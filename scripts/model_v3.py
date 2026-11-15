"""模型阶段 v3：干净数据 + 已筛选因子 + DoubleEnsemble + 概率校准 + 多空回测。
from stellarq.paths import DATA, ROOT, WIDE  # noqa: E402

与 v2 的差别：
1. 数据用 wide2（已过涨跌停合规闸门，剔除 ST、成交额<2000 万）；
2. 特征只用因子检验里 |ICIR|>0.25 且 BH q<0.01 的因子（防止噪声特征稀释）；
3. 标签 T+1 开盘买、T+6 开盘卖；训练/验证/测试按时间 70/15/15 切分；
4. 报告样本外 RankIC、校准 Brier/ECE，以及重叠修正后的多空年化。
"""
import sys, time, gc, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'model_v3.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'model_v3.log','w').close()
try:
    import numpy as np, pandas as pd
    import stellarq.factors.alpha_fast as af
    from stellarq.validation.fast import cs_prep
    from stellarq.models.ensemble import DoubleEnsembleRegressor, DEConfig, platt_fit, platt_apply, brier, ece, reliability
    from stellarq.models.gbdt import GBDTParams
    from stellarq.portfolio.costs import CostModel, stamp_tax_rate
    t0=time.time()
    R='WIDE/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o,h,l,c,v,amt,turn,cap = M('open'),M('high'),M('low'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap')
    check=M('avg_price')
    valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy', allow_pickle=True); syms=np.load(R+'wide_syms.npy', allow_pickle=True)
    nd,ns=c.shape
    H=6
    lab=np.full((nd,ns), np.nan, dtype=np.float32)
    for i0 in range(0, nd-1-H, 600):
        i1=min(i0+600, nd-1-H)
        buy=np.asarray(o[i0+1:i1+1],dtype=np.float64); sell=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)
        lab[i0:i1]=sell/buy-1.0
    lab=np.where(valid, lab, np.nan)
    w(f'label ready {round(time.time()-t0,1)}s')
    # 特征：挑选的因子（与挖掘阶段同名同口径）
    FEATS=['VSTD20','VSTD10','AMT_MA5','VSUMD60','AMT_RATIO60','AMT_RATIO20','VMA60','MA60','MA20','CORR20','MA10','MIN5','MIN20','RET20','STD20','TURN5','KMID','LN_CAP']
    def feat_block(name, i0, i1):
        import re
        m=re.match(r'([A-Z_]+?)(\d+)$', name); base, ww = (m.group(1), int(m.group(2))) if m else (name, 0)
        s=max(0,i0-max(ww,20)); off=i0-s
        cc=np.asarray(c[s:i1],dtype=np.float64)
        if name=='KMID': return (np.asarray(c[i0:i1],dtype=np.float64)-np.asarray(o[i0:i1],dtype=np.float64))/np.asarray(o[i0:i1],dtype=np.float64)
        if name=='LN_CAP': return np.log1p(np.asarray(cap[i0:i1],dtype=np.float64))
        if base=='MA': return af.roll_mean(cc,ww,max(2,ww//2))[off:]/np.asarray(c[i0:i1],dtype=np.float64)-1.0
        if base=='STD':
            r=np.full_like(cc,np.nan); r[1:]=cc[1:]/cc[:-1]-1.0
            return af.roll_std(r,ww,max(2,ww//2))[off:]
        if base=='MIN':
            ext=np.asarray(l[s:i1],dtype=np.float64)
            return af.roll_min_ignore(ext,ww,max(2,ww//2))[off:]/np.asarray(c[i0:i1],dtype=np.float64)-1.0
        if base=='RET': return np.asarray(c[i0-ww:i1-ww],dtype=np.float64)/np.asarray(c[i0:i1],dtype=np.float64)-1.0 if i0>=ww else np.full((i1-i0,ns),np.nan)
        if base=='TURN': return af.roll_mean(np.asarray(turn[s:i1],dtype=np.float64),ww,max(2,ww//2))[off:]/100.0
        if base=='VMA':
            ext=np.asarray(v[s:i1],dtype=np.float64); cur=np.asarray(v[i0:i1],dtype=np.float64)
            return af.roll_mean(ext,ww,max(2,ww//2))[off:]/(cur+1e-12)-1.0
        if base=='AMT_MA': return af.roll_mean(np.log1p(np.asarray(amt[s:i1],dtype=np.float64)),ww,max(2,ww//2))[off:]
        if base=='AMT_RATIO':
            ext=np.log1p(np.asarray(amt[s:i1],dtype=np.float64)); return ext[off:]-af.roll_mean(ext,ww,max(2,ww//2))[off:]
        if base=='VSTD': return af.roll_std(np.log1p(np.asarray(v[s:i1],dtype=np.float64)),ww,max(2,ww//2))[off:]
        if base=='VSUMD':
            vv=np.asarray(v[s:i1],dtype=np.float64); dv=np.full_like(vv,np.nan); dv[1:]=np.diff(vv,axis=0)
            pos=af.roll_mean(np.where(np.isfinite(dv),np.maximum(dv,0),np.nan),ww,max(2,ww//2))[off:]
            neg=af.roll_mean(np.where(np.isfinite(dv),np.maximum(-dv,0),np.nan),ww,max(2,ww//2))[off:]
            return (pos-neg)/(pos+neg+1e-12)
        if base=='CORR':
            r=np.full_like(cc,np.nan); r[1:]=cc[1:]/cc[:-1]-1.0
            vv=np.asarray(v[s:i1],dtype=np.float64); vr=np.full_like(vv,np.nan); vr[1:]=vv[1:]/vv[:-1]-1.0
            return af.roll_corr(r,np.log1p(np.abs(vr)),ww,max(3,ww//3))[off:]
        raise KeyError(name)
    # 构建设计矩阵（逐截面、每截面最多 800 只，压低内存）
    Xs=[]; ys=[]; ds=[]; ss=[]
    rng=np.random.default_rng(11)
    for t in range(70, nd-H-1, 5):
        m=valid[t]&np.isfinite(lab[t])
        idx=np.where(m)[0]
        if len(idx)<300: continue
        if len(idx)>800: idx=rng.choice(idx,800,replace=False)
        cols=[]
        for f in FEATS:
            cols.append(feat_block(f,t,t+1)[0][idx])
        block=np.column_stack(cols)
        okm=np.all(np.isfinite(block),axis=1)
        if okm.sum()<200: continue
        Xs.append(cs_prep(block[okm])); ys.append(lab[t][idx][okm]); ds.append(np.full(int(okm.sum()),t)); ss.append(idx[okm])
        if t%200==0: w(f'design t={t} rows={int(okm.sum())} {round(time.time()-t0)}s')
    X=np.vstack(Xs).astype(np.float32); y=np.concatenate(ys).astype(np.float32); d=np.concatenate(ds); sidx=np.concatenate(ss)
    del Xs,ys,ds,ss; gc.collect()
    w(f'design {X.shape} {round(time.time()-t0)}s')
    uniq=np.unique(d)
    tr=d<uniq[int(len(uniq)*0.70)]; va=(d>=uniq[int(len(uniq)*0.70)])&(d<uniq[int(len(uniq)*0.85)]); te=d>=uniq[int(len(uniq)*0.85)]
    w(f'split {int(tr.sum())}/{int(va.sum())}/{int(te.sum())} test from {pd.Timestamp(dates[uniq[int(len(uniq)*0.85)]]).date()}')
    de=DoubleEnsembleRegressor(DEConfig(n_models=2, gbdt=GBDTParams(n_estimators=60,max_depth=4,min_child_samples=300,subsample=0.6,colsample=0.8,early_stopping_rounds=8)))
    de.fit(X[tr].astype(np.float64), y[tr].astype(np.float64), d[tr], FEATS,
           X[va].astype(np.float64), y[va].astype(np.float64), d[va])
    w(f'fit {round(time.time()-t0)}s {de.history}')
    def rel(yy,dd):
        out=np.zeros(len(yy))
        for u in np.unique(dd):
            m=dd==u; out[m]=(yy[m]>np.median(yy[m])).astype(float)
        return out
    sv=de.predict(X[va].astype(np.float64),FEATS); st=de.predict(X[te].astype(np.float64),FEATS)
    yv=rel(y[va],d[va]); yt=rel(y[te],d[te])
    A,B=platt_fit(sv,yv); pr=platt_apply(st,A,B)
    dte=d[te]; yte=y[te]
    ics=[float(np.corrcoef(pd.Series(st[dte==u]).rank(), pd.Series(yte[dte==u]).rank())[0,1]) for u in np.unique(dte)]
    # 多空回测（top/bottom 各 50，按得分）
    te_rows=np.where(te)[0]
    costs=CostModel()
    per=[]
    for u in np.unique(dte):
        sel=te_rows[dte==u]
        sc=pd.Series(st[dte==u], index=sidx[sel])
        uu=int(u)
        rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
        rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms)
        top=sc.sort_values(ascending=False).head(50).index; bot=sc.sort_values().head(50).index
        rl=float(rp.reindex(top).mean()); rs=float(rp.reindex(bot).mean())
        turn_=0.5
        cost=costs.buy_fee(turn_/2)+costs.sell_fee(turn_/2)+stamp_tax_rate(dates[uu])*turn_/2
        per.append({'date':str(pd.Timestamp(dates[uu]).date()),'gross':rl-rs,'net':rl-rs-cost})
    pf=pd.DataFrame(per)
    ov=pf['net'].rolling(5).mean().dropna()
    res={'design_rows':int(X.shape[0]),'features':FEATS,'train':int(tr.sum()),'valid':int(va.sum()),'test':int(te.sum()),
         'test_ic_mean':float(np.nanmean(ics)),'test_ic_n':int(np.isfinite(ics).sum()),
         'brier_cal':brier(pr,yt),'brier_raw':brier(platt_apply(st,1.,0.),yt),'ece':ece(pr,yt),
         'platt':[A,B],'rounds':de.history,'reliability':reliability(pr,yt).to_dict('records'),
         'ls_n':len(pf),'ls_start':pf['date'].iloc[0] if len(pf) else None,'ls_end':pf['date'].iloc[-1] if len(pf) else None,
         'ls_ann_net_overlap_adj':float(ov.mean()*10.08) if len(ov) else None,
         'ls_sharpe_overlap_adj':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)) if len(ov)>2 and ov.std(ddof=1)>0 else None,
         'ls_win_rate':float((pf['net']>0).mean()) if len(pf) else None}
    json.dump(res, open(OUT / 'model_v3.json','w'), ensure_ascii=False, indent=1, default=float)
    pf.to_csv(OUT / 'model_v3_ls.csv', index=False)
    w(json.dumps({k:res[k] for k in ('test_ic_mean','test_ic_n','brier_cal','ece','ls_ann_net_overlap_adj','ls_sharpe_overlap_adj','ls_win_rate','ls_start','ls_end')}, ensure_ascii=False))
    w('DONE '+str(round(time.time()-t0)))
except Exception:
    w(traceback.format_exc())
