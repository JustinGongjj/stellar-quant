"""v5 训练与回测：v4 量价特征 + 8 个基本面特征。"""
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
import sys, json, resource, gc, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    from stellarq.models.ensemble import DoubleEnsembleRegressor, DEConfig, platt_fit, platt_apply, brier, ece
    from stellarq.models.gbdt import GBDTParams
    from stellarq.portfolio.costs import CostModel, stamp_tax_rate
    C='CACHE/'
    FEATS_V4=json.load(open(OUT / '../data/model_v4_features.json')) if False else None
    V4=['KMID','LN_CAP','MA10','MA20','MA60','MA20_resid','MA60_resid','STD20','MIN5','MIN20','RET20','TURN5',
        'VSTD20','VSTD20_resid','AMT_RATIO60','AMT_RATIO60_resid','VMA60','VSUMD60','CORR20']
    F8=['f_bp','f_ep','f_growth','f_rev_grow','f_cfo','f_rev_acc','f_margin','f_roe']
    X=np.load(C+'v4_X.npy'); extra=np.load(C+'v5_extra.npy')
    y=np.load(C+'v4_y.npy'); d=np.load(C+'v4_d.npy'); sidx=np.load(C+'v4_s.npy')
    # 基本面缺失 → 用该日截面中位数填充（不丢样本）
    E=extra.copy(); del extra; gc.collect()
    for ui,u in enumerate(np.unique(d)):
        m=d==u
        for j in range(E.shape[1]):
            col=E[m,j]; med=np.nanmedian(col)
            col[~np.isfinite(col)]=med if np.isfinite(med) else 0.0
            E[m,j]=col
    X5=np.hstack([X, E]).astype(np.float32); del X, E; gc.collect()
    FEATS=V4+F8
    w(f'design {X5.shape}')
    uniq=np.unique(d); cut=uniq[int(len(uniq)*0.85)]
    tr=d<uniq[int(len(uniq)*0.70)]; va=(d>=uniq[int(len(uniq)*0.70)])&(d<cut)
    de=DoubleEnsembleRegressor(DEConfig(n_models=2,gbdt=GBDTParams(n_estimators=60,max_depth=4,min_child_samples=300,subsample=0.6,colsample=0.8,early_stopping_rounds=8)))
    de.fit(X5[tr].astype(np.float64),y[tr].astype(np.float64),d[tr],FEATS,X5[va].astype(np.float64),y[va].astype(np.float64),d[va])
    w(f'fit {de.history}')
    st=de.predict(X5[d>=cut].astype(np.float64),FEATS); dte=d[d>=cut]; ste=sidx[d>=cut]
    ya=y[d>=cut]
    ics=[float(np.corrcoef(pd.Series(st[dte==u]).rank(), pd.Series(ya[dte==u]).rank())[0,1]) for u in np.unique(dte)]
    def rel(yy,dd):
        out=np.zeros(len(yy))
        for u in np.unique(dd):
            mm=dd==u; out[mm]=(yy[mm]>np.median(yy[mm])).astype(float)
        return out
    sv=de.predict(X5[va].astype(np.float64),FEATS); yv=rel(y[va],d[va]); yt=rel(ya,dte)
    A,B=platt_fit(sv,yv); pr=platt_apply(st,A,B)
    np.save(C+'v5_st.npy',st); np.save(C+'v5_dte.npy',dte); np.save(C+'v5_ste.npy',ste)
    res={'features':FEATS,'design':list(X5.shape),'test_ic':float(np.nanmean(ics)),'test_ic_n':int(np.isfinite(ics).sum()),
         'brier_cal':brier(pr,yt),'brier_raw':brier(platt_apply(st,1.,0.),yt),'ece':ece(pr,yt),'platt':[A,B],'rounds':de.history}
    json.dump(res, open(OUT / 'model_v5.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps({k:res[k] for k in ('test_ic','test_ic_n','ece')}, ensure_ascii=False))
    w('DONE')
except Exception:
    w(traceback.format_exc())
