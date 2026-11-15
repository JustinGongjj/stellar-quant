"""模型 v6：v4 的量价 19 特征 + f_bp/f_rev_grow（样本外仍有效的 2 个基本面）
from stellarq.paths import CACHE, DATA, ROOT, WIDE  # noqa: E402
                + ev_abnormal_20d（异常波动公告，样本外确认有效）。
按 phase8 的诊断结论剔除 f_roe/f_margin/f_ep/f_growth/f_cfo/f_rev_acc。
"""
import sys, glob, json, gc, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    from stellarq.factors.event_fund import event_matrix
    from stellarq.models.ensemble import DoubleEnsembleRegressor, DEConfig, platt_fit, platt_apply, brier, ece
    from stellarq.models.gbdt import GBDTParams
    from stellarq.portfolio.costs import CostModel, stamp_tax_rate
    C="CACHE/"
    H=6
    V4=['KMID','LN_CAP','MA10','MA20','MA60','MA20_resid','MA60_resid','STD20','MIN5','MIN20','RET20','TURN5',
        'VSTD20','VSTD20_resid','AMT_RATIO60','AMT_RATIO60_resid','VMA60','VSUMD60','CORR20']
    X=np.load(C+'v4_X.npy'); extra=np.load(C+'v5_extra.npy')
    y=np.load(C+'v4_y.npy'); d=np.load(C+'v4_d.npy'); sidx=np.load(C+'v4_s.npy')
    BCOLS=[0,3]   # f_bp, f_rev_grow
    B=extra[:,BCOLS]; del extra; gc.collect()
    for u in np.unique(d):
        m=d==u
        for j in range(B.shape[1]):
            col=B[m,j]; med=np.nanmedian(col)
            col[~np.isfinite(col)]=med if np.isfinite(med) else 0.0
            B[m,j]=col
    w('fundamentals aligned')
    R='WIDE/'
    dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
    ev=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(C+'events/*.parquet'))], ignore_index=True)
    ev=ev.drop_duplicates(subset=['symbol','ann_date','event'])
    EM=event_matrix(ev, dates, syms); del ev; gc.collect()
    ab=EM['abnormal']
    gath=np.asarray(ab[d, sidx], dtype=np.float32).reshape(-1,1)
    del EM, ab; gc.collect()
    w('events gathered')
    FEATS=V4+['f_bp','f_rev_grow','ev_abnormal_20d']
    X6=np.hstack([X, B, gath]).astype(np.float32)
    del X, B, gath; gc.collect()
    w(f'design {X6.shape}')
    uniq=np.unique(d); cut=uniq[int(len(uniq)*0.85)]
    tr=d<uniq[int(len(uniq)*0.70)]; va=(d>=uniq[int(len(uniq)*0.70)])&(d<cut)
    de=DoubleEnsembleRegressor(DEConfig(n_models=2,gbdt=GBDTParams(n_estimators=60,max_depth=4,min_child_samples=300,subsample=0.6,colsample=0.8,early_stopping_rounds=8)))
    de.fit(X6[tr].astype(np.float64),y[tr].astype(np.float64),d[tr],FEATS,X6[va].astype(np.float64),y[va].astype(np.float64),d[va])
    w(f'fit {de.history}')
    st=de.predict(X6[d>=cut].astype(np.float64),FEATS); dte=d[d>=cut]; ste=sidx[d>=cut]; ya=y[d>=cut]
    ics=[float(np.corrcoef(pd.Series(st[dte==u]).rank(), pd.Series(ya[dte==u]).rank())[0,1]) for u in np.unique(dte)]
    def rel(yy,dd):
        out=np.zeros(len(yy))
        for u in np.unique(dd):
            mm=dd==u; out[mm]=(yy[mm]>np.median(yy[mm])).astype(float)
        return out
    sv=de.predict(X6[va].astype(np.float64),FEATS); yv=rel(y[va],d[va]); yt=rel(ya,dte)
    A,Bp=platt_fit(sv,yv); pr=platt_apply(st,A,Bp)
    # 多头端回测（缓冲区 60/80，top50，含冲击成本）
    o=np.load(R+'wide_open.npy', mmap_mode='r'); amt=np.load(R+'wide_amount.npy', mmap_mode='r'); valid=np.load(R+'wide_valid.npy')
    COMM=0.00025; TRANS=0.00001; SLIP=0.0005
    def stamp(dt): return 0.0005 if pd.Timestamp(dt)>=pd.Timestamp('2023-08-28') else 0.001
    dl=list(np.unique(dte)); sc_by_date={u: pd.Series(st[dte==u], index=[str(syms[i]) for i in ste[dte==u]]) for u in dl}
    prev=set(); per=[]
    for u in dl:
        uu=int(u); sc=sc_by_date[u]
        ranked=sc.sort_values(ascending=False); rank={s:i for i,s in enumerate(ranked.index)}
        keep=[s for s in prev if rank.get(s,10**9)<80]
        held=list(dict.fromkeys(keep+list(ranked.head(60).index)))[:50]
        rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
        rp=pd.Series(np.where(valid[uu],rp,np.nan), index=[str(s) for s in syms])
        r=rp.reindex(held).dropna()
        if len(r)<40: continue
        adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=[str(s) for s in syms]).reindex(held)
        impact=float(np.nanmean(np.minimum(0.003,(1e8/len(held))/np.maximum(adv.values,1e4)*0.1)))
        new=set(held); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
        cost=turn_*(COMM+TRANS+SLIP+impact)+turn_*0.5*stamp(dates[uu])
        per.append({'gross':float(r.mean()),'net':float(r.mean())-cost,'bench':float(rp.dropna().mean()),'turn':turn_})
    pf=pd.DataFrame(per)
    ov=pf['net'].rolling(5).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(5).mean().dropna()
    res={'features':FEATS,'design':list(X6.shape),'test_ic':float(np.nanmean(ics)),'test_ic_n':int(np.isfinite(ics).sum()),
         'ece':ece(pr,yt),'rounds':de.history,
         'long_only':{'n':int(len(pf)),'ann_net':float(ov.mean()*10.08),'ann_excess':float(ex.mean()*10.08),
                      'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),'turnover':float(pf['turn'].mean())}}
    json.dump(res, open(OUT / 'model_v6.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps({'test_ic':res['test_ic'],'ece':res['ece'],'long_only':res['long_only']}, ensure_ascii=False))
    w('DONE')
except Exception:
    w(traceback.format_exc())
