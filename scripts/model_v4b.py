"""从缓存的设计矩阵训练 + 回测（避免重复构造，内存可控）。"""
from stellarq.paths import CACHE, DATA, ROOT, WIDE  # noqa: E402
import sys, time, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'model_v4.log','a') as f: f.write(str(m)+'\n')
try:
    import numpy as np, pandas as pd, gc
    from stellarq.models.ensemble import DoubleEnsembleRegressor, DEConfig, platt_fit, platt_apply, brier, ece
    from stellarq.models.gbdt import GBDTParams
    from stellarq.portfolio.costs import CostModel, stamp_tax_rate
    t0=time.time()
    C='CACHE/'
    X=np.load(C+'v4_X.npy'); y=np.load(C+'v4_y.npy'); d=np.load(C+'v4_d.npy'); sidx=np.load(C+'v4_s.npy')
    R='WIDE/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
    nd,ns=o.shape; H=6
    FEATS=['KMID','LN_CAP','MA10','MA20','MA60','MA20_resid','MA60_resid','STD20','MIN5','MIN20','RET20','TURN5',
           'VSTD20','VSTD20_resid','AMT_RATIO60','AMT_RATIO60_resid','VMA60','VSUMD60','CORR20']
    w(f'loaded {X.shape} {round(time.time()-t0)}s')
    uniq=np.unique(d); cut=uniq[int(len(uniq)*0.85)]
    tr=d<uniq[int(len(uniq)*0.70)]; va=(d>=uniq[int(len(uniq)*0.70)])&(d<cut)
    de=DoubleEnsembleRegressor(DEConfig(n_models=2,gbdt=GBDTParams(n_estimators=60,max_depth=4,min_child_samples=300,subsample=0.6,colsample=0.8,early_stopping_rounds=8)))
    de.fit(X[tr].astype(np.float64),y[tr].astype(np.float64),d[tr],FEATS,X[va].astype(np.float64),y[va].astype(np.float64),d[va])
    w(f'fit {round(time.time()-t0)}s {de.history}')
    st=de.predict(X[d>=cut].astype(np.float64),FEATS); dte=d[d>=cut]; ste=sidx[d>=cut]
    np.save(C+'v4_st.npy',st); np.save(C+'v4_dte.npy',dte); np.save(C+'v4_ste.npy',ste)
    def rel(yy,dd):
        out=np.zeros(len(yy))
        for u in np.unique(dd):
            mm=dd==u; out[mm]=(yy[mm]>np.median(yy[mm])).astype(float)
        return out
    sv=de.predict(X[va].astype(np.float64),FEATS); yv=rel(y[va],d[va]); yt=rel(y[d>=cut],dte)
    A,B=platt_fit(sv,yv); pr=platt_apply(st,A,B)
    ics=[float(np.corrcoef(pd.Series(st[dte==u]).rank(), pd.Series(y[d>=cut][dte==u]).rank())[0,1]) for u in np.unique(dte)]
    costs=CostModel()
    date_list=list(np.unique(dte))
    sc_by_date={u: pd.Series(st[dte==u], index=[str(syms[i]) for i in ste[dte==u]]) for u in date_list}
    def run(buf_in, buf_out, smooth, cap=50):
        prev=set(); per=[]
        for ui,u in enumerate(date_list):
            uu=int(u); sc=sc_by_date[u]
            if smooth and ui>=2:
                sc=(sc.add(sc_by_date[date_list[ui-1]],fill_value=np.nan)
                      .add(sc_by_date[date_list[ui-2]],fill_value=np.nan))/3.0
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=[str(s) for s in syms])
            ranked=sc.sort_values(ascending=False)
            rank={s:i for i,s in enumerate(ranked.index)}
            top=list(ranked.head(buf_in).index)
            keep=[s for s in prev if rank.get(s,10**9)<buf_out] if buf_out else []
            held=list(dict.fromkeys(keep+top))[:cap]
            bottom=list(sc.sort_values().head(50).index)
            if rp.reindex(held).notna().mean()<0.8 or rp.reindex(bottom).notna().mean()<0.8: continue
            rl=float(rp.reindex(held).mean()); rs=float(rp.reindex(bottom).mean())
            new=set(held); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
            cost=costs.buy_fee(turn_/2)+costs.sell_fee(turn_/2)+stamp_tax_rate(dates[uu])*turn_/2
            per.append({'gross':rl-rs,'net':rl-rs-cost,'turnover':turn_})
        pf=pd.DataFrame(per); ov=pf['net'].rolling(5).mean().dropna()
        return {'n':int(len(pf)),'turnover':float(pf['turnover'].mean()),
                'ann_net':float(ov.mean()*10.08),'sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
                'ann_gross':float(pf['gross'].mean()*50.4),'win':float((pf['net']>0).mean())}
    bt={'baseline_50':run(50,0,False), 'smooth3_50':run(50,0,True),
        'buffer_60_80':run(60,80,False), 'smooth3_buffer_60_80':run(60,80,True),
        'smooth3_buffer_80_120':run(80,120,True)}
    res={'design_rows':int(X.shape[0]),'features':FEATS,'test_ic':float(np.nanmean(ics)),'test_ic_n':int(np.isfinite(ics).sum()),
         'brier_cal':brier(pr,yt),'brier_raw':brier(platt_apply(st,1.,0.),yt),'ece':ece(pr,yt),'platt':[A,B],
         'rounds':de.history,'backtests':bt}
    json.dump(res, open(OUT / 'model_v4.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps({'test_ic':res['test_ic'],'ece':res['ece'],'bt':bt}, ensure_ascii=False, indent=1, default=float))
    w('DONE '+str(round(time.time()-t0)))
except Exception:
    w(traceback.format_exc())
