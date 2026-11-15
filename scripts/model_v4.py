"""模型 v4：残差化因子 + 换手抑制，对比 v3。
from stellarq.paths import CACHE, DATA, ROOT, WIDE  # noqa: E402

三项改动：
1. 因子集里把 VSTD20/AMT_RATIO60/MA60 换成残差化版本（剔除 ln市值/60日波动/换手）；
2. 打分做 3 期移动平均平滑（等价「分多次建仓」）；
3. 用「缓冲区」选股：排名进前 60 才买、跌出前 80 才卖，降低换手。
输出：RankIC、Brier/ECE、换手、毛/净收益（重叠修正）。
"""
import sys, time, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'model_v4.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'model_v4.log','w').close()
try:
    import numpy as np, pandas as pd
    import stellarq.factors.alpha_fast as af
    from stellarq.validation.fast import cs_prep
    from stellarq.models.ensemble import DoubleEnsembleRegressor, DEConfig, platt_fit, platt_apply, brier, ece
    from stellarq.models.gbdt import GBDTParams
    from stellarq.portfolio.costs import CostModel, stamp_tax_rate
    t0=time.time()
    R='WIDE/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o,c,v,amt,turn,cap,l = M('open'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap'),M('low')
    valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
    nd,ns=c.shape; H=6
    lab=np.full((nd,ns),np.nan,dtype=np.float32)
    for i0 in range(0,nd-1-H,600):
        i1=min(i0+600,nd-1-H)
        lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
    lab=np.where(valid,lab,np.nan)
    w(f'label {round(time.time()-t0)}s')

    def roll_mean_f(x,k):
        return af.roll_mean(x,k,max(2,k//2))
    def roll_std_f(x,k):
        return af.roll_std(x,k,max(2,k//2))

    def raw_feats(t):
        """在 t 处算全部基础特征（向量化，一次取窗口）。"""
        s=max(0,t-65); off=t-s
        cc=np.asarray(c[s:t+1],dtype=np.float64)
        ret=np.full_like(cc,np.nan); ret[1:]=cc[1:]/cc[:-1]-1.0
        vv=np.asarray(v[s:t+1],dtype=np.float64); aa=np.asarray(amt[s:t+1],dtype=np.float64)
        tt=np.asarray(turn[s:t+1],dtype=np.float64); ll=np.asarray(l[s:t+1],dtype=np.float64)
        cur_c=np.asarray(c[t],dtype=np.float64); cur_o=np.asarray(o[t],dtype=np.float64)
        lv=np.log1p(vv); la=np.log1p(aa)
        f={}
        f['KMID']=(cur_c-cur_o)/cur_o
        f['LN_CAP']=np.log1p(np.asarray(cap[t],dtype=np.float64))
        f['MA60']=roll_mean_f(cc,60)[off]/cur_c-1.0
        f['MA20']=roll_mean_f(cc,20)[off]/cur_c-1.0
        f['MA10']=roll_mean_f(cc,10)[off]/cur_c-1.0
        f['STD20']=roll_std_f(ret,20)[off]
        f['MIN5']=af.roll_min_ignore(ll,5,2)[off]/cur_c-1.0
        f['MIN20']=af.roll_min_ignore(ll,20,10)[off]/cur_c-1.0
        f['RET20']=cc[off-20]/cur_c-1.0
        f['TURN5']=roll_mean_f(tt,5)[off]/100.0
        f['VSTD20']=roll_std_f(lv,20)[off]
        f['AMT_RATIO60']=la[off]-roll_mean_f(la,60)[off]
        f['VMA60']=roll_mean_f(vv,60)[off]/(np.asarray(v[t],dtype=np.float64)+1e-12)-1.0
        f['VSUMD60']=(roll_mean_f(np.where(np.isfinite(np.vstack([np.full((1,ns),np.nan),np.diff(vv,axis=0)])),np.maximum(np.vstack([np.full((1,ns),np.nan),np.diff(vv,axis=0)]),0),np.nan),60)[off]
                      -roll_mean_f(np.where(np.isfinite(np.vstack([np.full((1,ns),np.nan),np.diff(vv,axis=0)])),np.maximum(-np.vstack([np.full((1,ns),np.nan),np.diff(vv,axis=0)]),0),np.nan),60)[off])
        vr=np.full_like(vv,np.nan); vr[1:]=vv[1:]/vv[:-1]-1.0
        f['CORR20']=af.roll_corr(ret,np.log1p(np.abs(vr)),20,7)[off]
        # 残差化所需的暴露
        vol60=roll_std_f(ret,60)[off]
        # 用 20 日成交量/成交额波动作为 VSTD20 / AMT_RATIO60 的残差基础
        X=np.column_stack([np.ones(ns),f['LN_CAP'],vol60,f['TURN5']])
        def resid(y):
            out=np.full(ns,np.nan)
            m=np.isfinite(y)&np.all(np.isfinite(X),axis=1)
            if m.sum()<200: return out
            b,*_=np.linalg.lstsq(X[m],y[m],rcond=None)
            out[m]=y[m]-X[m]@b
            return out
        r20=resid(f['MA20']); f['MA20_resid']=r20
        r60=resid(f['MA60']); f['MA60_resid']=r60
        rv=resid(f['VSTD20']); f['VSTD20_resid']=rv
        ra=resid(f['AMT_RATIO60']); f['AMT_RATIO60_resid']=ra
        return f

    FEATS=['KMID','LN_CAP','MA10','MA20','MA60','MA20_resid','MA60_resid','STD20','MIN5','MIN20','RET20','TURN5',
           'VSTD20','VSTD20_resid','AMT_RATIO60','AMT_RATIO60_resid','VMA60','VSUMD60','CORR20']
    rows=[]
    rng=np.random.default_rng(23)
    sid_hash=np.array([abs(hash(str(s))) for s in syms], dtype=np.int64)
    for t in range(70,nd-H-1,5):
        m=valid[t]&np.isfinite(lab[t]); idx=np.where(m)[0]
        if len(idx)<300: continue
        # 重要：抽样必须对股票「固定」，否则相邻截面股票池不同（实测重合仅 24%），
        # 后续的时序平滑就变成比较不同股票，回测会崩掉。
        if len(idx)>800:
            keep_mask=(sid_hash[idx]% 5)==0
            idx=idx[keep_mask] if keep_mask.sum()>=300 else idx[rng.choice(len(idx),800,replace=False)]
        f=raw_feats(t)
        Mx=np.column_stack([f[k][idx] for k in FEATS])
        okm=np.all(np.isfinite(Mx),axis=1)
        if okm.sum()<200: continue
        rows.append((t, idx[okm], cs_prep(Mx[okm]), lab[t][idx[okm]]))
    X=np.vstack([r[2] for r in rows]).astype(np.float32)
    y=np.concatenate([r[3] for r in rows]).astype(np.float32)
    d=np.concatenate([np.full(len(r[1]),r[0]) for r in rows])
    sidx=np.concatenate([r[1] for r in rows])
    w(f'design {X.shape} {round(time.time()-t0)}s')
    np.save('CACHE/v4_X.npy', X); np.save('CACHE/v4_y.npy', y)
    np.save('CACHE/v4_d.npy', d); np.save('CACHE/v4_s.npy', sidx)
    uniq=np.unique(d); cut=uniq[int(len(uniq)*0.85)]
    tr=d<uniq[int(len(uniq)*0.70)]; va=(d>=uniq[int(len(uniq)*0.70)])&(d<cut)
    de=DoubleEnsembleRegressor(DEConfig(n_models=2,gbdt=GBDTParams(n_estimators=60,max_depth=4,min_child_samples=300,subsample=0.6,colsample=0.8,early_stopping_rounds=8)))
    de.fit(X[tr].astype(np.float64),y[tr].astype(np.float64),d[tr],FEATS,X[va].astype(np.float64),y[va].astype(np.float64),d[va])
    w(f'fit {round(time.time()-t0)}s {de.history}')
    st=de.predict(X[d>=cut].astype(np.float64),FEATS); dte=d[d>=cut]; ste=sidx[d>=cut]
    np.save('CACHE/v4_st.npy', st); np.save('CACHE/v4_dte.npy', dte); np.save('CACHE/v4_ste.npy', ste)
    def rel(yy,dd):
        out=np.zeros(len(yy))
        for u in np.unique(dd):
            mm=dd==u; out[mm]=(yy[mm]>np.median(yy[mm])).astype(float)
        return out
    sv=de.predict(X[va].astype(np.float64),FEATS); yv=rel(y[va],d[va]); yt=rel(y[d>=cut],dte)
    A,B=platt_fit(sv,yv); pr=platt_apply(st,A,B)
    ics=[float(np.corrcoef(pd.Series(st[dte==u]).rank(), pd.Series(y[d>=cut][dte==u]).rank())[0,1]) for u in np.unique(dte)]
    # ---- 回测：原始 vs 缓冲区
    costs=CostModel(); out_ls={}
    date_list=list(np.unique(dte)); pos={u:i for i,u in enumerate(date_list)}
    sc_by_date={u: pd.Series(st[dte==u], index=[str(syms[i]) for i in ste[dte==u]]) for u in date_list}
    def run(buf_in, buf_out, smooth):
        prev=set(); per=[]
        for u in date_list:
            uu=int(u)
            sc=sc_by_date[u]
            if smooth:
                prev_dates=[x for x in date_list if x<uu][-2:]
                if len(prev_dates)==2:
                    sc=(sc.add(sc_by_date[prev_dates[0]],fill_value=np.nan)
                          .add(sc_by_date[prev_dates[1]],fill_value=np.nan))/3.0
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=[str(s) for s in syms])
            ranked=sc.sort_values(ascending=False)
            rank={s:i for i,s in enumerate(ranked.index)}
            top=list(ranked.head(buf_in).index)
            keep=[s for s in prev if rank.get(s,10**9)<buf_out] if buf_out else []
            held=list(dict.fromkeys(keep+top))[:50]
            bottom=list(sc.sort_values().head(50).index)
            if rp.reindex(held).notna().mean()<0.8 or rp.reindex(bottom).notna().mean()<0.8: continue
            rl=float(rp.reindex(held).mean()); rs=float(rp.reindex(bottom).mean())
            new=set(held); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
            cost=costs.buy_fee(turn_/2)+costs.sell_fee(turn_/2)+stamp_tax_rate(dates[uu])*turn_/2
            per.append({'gross':rl-rs,'net':rl-rs-cost,'turnover':turn_})
        pf=pd.DataFrame(per)
        ov=pf['net'].rolling(5).mean().dropna()
        return {'n':len(pf),'turnover':float(pf['turnover'].mean()),
                'ann_net_overlap_adj':float(ov.mean()*10.08),'sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
                'ann_gross':float(pf['gross'].mean()*50.4),'win':float((pf['net']>0).mean())}
    out_ls['baseline']=run(50,0,False)
    out_ls['smooth']=run(50,0,True)
    out_ls['buffer']=run(60,80,False)
    out_ls['smooth_buffer']=run(60,80,True)
    res={'design_rows':int(X.shape[0]),'features':FEATS,'test_ic':float(np.nanmean(ics)),'test_ic_n':int(np.isfinite(ics).sum()),
         'brier_cal':brier(pr,yt),'brier_raw':brier(platt_apply(st,1.,0.),yt),'ece':ece(pr,yt),'platt':[A,B],
         'rounds':de.history,'backtests':out_ls}
    json.dump(res, open(OUT / 'model_v4.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps({'test_ic':res['test_ic'],'ece':res['ece'],'backtests':out_ls}, ensure_ascii=False, indent=1, default=float))
    w('DONE '+str(round(time.time()-t0)))
except Exception:
    w(traceback.format_exc())
