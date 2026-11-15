"""多空回测（独立于模型训练，复用 model_v3.json 的样本外得分区间）。
from stellarq.paths import DATA, ROOT, WIDE  # noqa: E402

修正点：上一版把「得分索引 = 设计矩阵行」误当成「股票索引」，导致 rp 全 NaN。
这里直接用 sidx 映射回股票，并显式检查覆盖率。
"""
import sys, time, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'backtest_v3.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'backtest_v3.log','w').close()
try:
    import numpy as np, pandas as pd
    import stellarq.factors.alpha_fast as af
    from stellarq.validation.fast import cs_prep
    from stellarq.models.ensemble import DoubleEnsembleRegressor, DEConfig, platt_fit, platt_apply
    from stellarq.models.gbdt import GBDTParams
    from stellarq.portfolio.costs import CostModel, stamp_tax_rate
    t0=time.time()
    R='WIDE/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o,c,v,amt,turn,cap,l = M('open'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap'),M('low')
    valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
    nd,ns=c.shape; H=6
    FEATS=['VSTD20','VSTD10','AMT_MA5','VSUMD60','AMT_RATIO60','AMT_RATIO20','VMA60','MA60','MA20','CORR20','MA10','MIN5','MIN20','RET20','STD20','TURN5','KMID','LN_CAP']
    def feat_block(name,i0,i1):
        import re
        m=re.match(r'([A-Z_]+?)(\d+)$',name); base,ww=(m.group(1),int(m.group(2))) if m else (name,0)
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
        if base=='RET': return np.asarray(c[i0-ww:i1-ww],dtype=np.float64)/np.asarray(c[i0:i1],dtype=np.float64)-1.0
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
    lab=np.full((nd,ns),np.nan,dtype=np.float32)
    for i0 in range(0,nd-1-H,600):
        i1=min(i0+600,nd-1-H)
        lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
    lab=np.where(valid,lab,np.nan)
    Xs=[];ys=[];ds=[];ss=[]; rng=np.random.default_rng(11)
    for t in range(70,nd-H-1,5):
        m=valid[t]&np.isfinite(lab[t]); idx=np.where(m)[0]
        if len(idx)<300: continue
        if len(idx)>800: idx=rng.choice(idx,800,replace=False)
        block=np.column_stack([feat_block(f,t,t+1)[0][idx] for f in FEATS])
        okm=np.all(np.isfinite(block),axis=1)
        if okm.sum()<200: continue
        Xs.append(cs_prep(block[okm])); ys.append(lab[t][idx][okm]); ds.append(np.full(int(okm.sum()),t)); ss.append(idx[okm])
    X=np.vstack(Xs).astype(np.float32); y=np.concatenate(ys).astype(np.float32); d=np.concatenate(ds); sidx=np.concatenate(ss)
    del Xs,ys,ds,ss
    uniq=np.unique(d); cut=uniq[int(len(uniq)*0.85)]
    tr=d<uniq[int(len(uniq)*0.70)]; va=(d>=uniq[int(len(uniq)*0.70)])&(d<cut)
    de=DoubleEnsembleRegressor(DEConfig(n_models=2,gbdt=GBDTParams(n_estimators=60,max_depth=4,min_child_samples=300,subsample=0.6,colsample=0.8,early_stopping_rounds=8)))
    de.fit(X[tr].astype(np.float64),y[tr].astype(np.float64),d[tr],FEATS,X[va].astype(np.float64),y[va].astype(np.float64),d[va])
    st_all=de.predict(X[d>=cut].astype(np.float64),FEATS)
    dte=d[d>=cut]; sidx_te=sidx[d>=cut]
    w(f'scored {len(st_all)} rows on {len(np.unique(dte))} dates, {round(time.time()-t0)}s')
    costs=CostModel(); per=[]; prev=set()
    for u in np.unique(dte):
        rows=np.where(dte==u)[0]
        if len(rows)<100: continue
        uu=int(u)
        sc=pd.Series(st_all[rows], index=[str(syms[i]) for i in sidx_te[rows]])
        rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
        rp=pd.Series(np.where(valid[uu],rp,np.nan), index=[str(s) for s in syms])
        top=list(sc.sort_values(ascending=False).head(50).index); bot=list(sc.sort_values().head(50).index)
        cover=float(rp.reindex(top+bot).notna().mean())
        if cover<0.8: continue
        rl=float(rp.reindex(top).mean()); rs=float(rp.reindex(bot).mean())
        new=set(top); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
        cost=costs.buy_fee(turn_/2)+costs.sell_fee(turn_/2)+stamp_tax_rate(dates[uu])*turn_/2
        per.append({'date':str(pd.Timestamp(dates[uu]).date()),'gross':rl-rs,'net':rl-rs-cost,'turnover':turn_,'cover':cover})
    pf=pd.DataFrame(per)
    ov=pf['net'].rolling(5).mean().dropna()
    res={'n_periods':len(pf),'start':pf['date'].iloc[0],'end':pf['date'].iloc[-1],
         'ann_gross_naive':float(pf['gross'].mean()*50.4),'ann_net_naive':float(pf['net'].mean()*50.4),
         'ann_net_overlap_adj':float(ov.mean()*10.08),'sharpe_overlap_adj':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
         'mean_turnover':float(pf['turnover'].mean()),'win_rate':float((pf['net']>0).mean()),
         'worst':float(pf['net'].min()),'best':float(pf['net'].max()),
         'by_year':{str(k):float(v) for k,v in pf.assign(y=pf['date'].str[:4]).groupby('y')['net'].sum().items()}}
    json.dump(res, open(OUT / 'backtest_v3.json','w'), ensure_ascii=False, indent=1, default=float)
    pf.to_csv(OUT / 'backtest_v3.csv', index=False)
    w(json.dumps(res, ensure_ascii=False))
except Exception:
    w(traceback.format_exc())
