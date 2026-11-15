"""模型 v5：在 v4（量价 19 因子）基础上加入基本面因子，看信号能否上台阶。
from stellarq.paths import CACHE, DATA, ROOT, WIDE  # noqa: E402

新增特征（全部按公告日 point-in-time 对齐，无前视）：
  f_bp（账面市值比）、f_ep（盈利收益率）、f_growth（净利同比）、f_rev_grow（营收同比）、
  f_cfo（现金流收益率）、f_rev_acc（营收同比−净利同比）、f_margin、f_roe
并与 v4 做同口径对比（同切分、同回测、同费用）。
"""
import sys, time, json, traceback, resource; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    print(str(m), f'[mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import glob, gc, numpy as np, pandas as pd
    import stellarq.factors.alpha_fast as af
    from stellarq.validation.fast import cs_prep
    from stellarq.factors.event_fund import fundamental_panel, point_in_time
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
    # 基本面（point-in-time）
    files=sorted(glob.glob('CACHE/fund/*.parquet'))
    fp=fundamental_panel(files, pd.Timestamp(dates[-1]))
    mat=point_in_time(fp, dates, syms, ['WEIGHTAVG_ROE','SJLTZ','YSTZ','XSMLL','BASIC_EPS','BPS','MGJYXJJE'])
    del fp; gc.collect()
    close=np.asarray(c,dtype=np.float32)
    FUND={}
    def _make(name, num, den=None):
        """逐列原地生成，避免同时分配 7 个 (3581×5027) 中间数组（实测会 OOM）。"""
        out=np.array(num, dtype=np.float32, copy=True)
        if den is not None:
            bad = ~(den > 0)
            out[bad] = np.nan          # 原地置 NaN，避免额外分配一个全尺寸数组
            out /= den
        out[~valid]=np.nan
        FUND[name]=out
    close_safe=np.array(close, dtype=np.float32, copy=True)
    _make('f_bp', mat['BPS'], close_safe)
    _make('f_ep', mat['BASIC_EPS'], close_safe)
    _make('f_growth', mat['SJLTZ'])
    _make('f_rev_grow', mat['YSTZ'])
    _make('f_cfo', mat['MGJYXJJE'], close_safe)
    _make('f_margin', mat['XSMLL'])
    _make('f_roe', mat['WEIGHTAVG_ROE'])
    acc=np.array(mat['YSTZ'], dtype=np.float32, copy=True); acc-=mat['SJLTZ']; acc[~valid]=np.nan
    FUND['f_rev_acc']=acc
    del close_safe, acc, close; gc.collect()
    w(f'fund ready {round(time.time()-t0)}s')
    FEATS=['KMID','LN_CAP','MA10','MA20','MA60','MA20_resid','MA60_resid','STD20','MIN5','MIN20','RET20','TURN5',
           'VSTD20','VSTD20_resid','AMT_RATIO60','AMT_RATIO60_resid','VMA60','VSUMD60','CORR20']+list(FUND)
    def roll_mean_f(x,k): return af.roll_mean(x,k,max(2,k//2))
    def roll_std_f(x,k): return af.roll_std(x,k,max(2,k//2))
    def base_feats(t):
        s=max(0,t-65); off=t-s
        cc=np.asarray(c[s:t+1],dtype=np.float64)
        ret=np.full_like(cc,np.nan); ret[1:]=cc[1:]/cc[:-1]-1.0
        vv=np.asarray(v[s:t+1],dtype=np.float64); aa=np.asarray(amt[s:t+1],dtype=np.float64)
        tt=np.asarray(turn[s:t+1],dtype=np.float64); ll=np.asarray(l[s:t+1],dtype=np.float64)
        cur_c=np.asarray(c[t],dtype=np.float64); cur_o=np.asarray(o[t],dtype=np.float64)
        lv=np.log1p(vv); la=np.log1p(aa)
        f={}
        f['KMID']=(cur_c-cur_o)/cur_o; f['LN_CAP']=np.log1p(np.asarray(cap[t],dtype=np.float64))
        f['MA60']=roll_mean_f(cc,60)[off]/cur_c-1.0; f['MA20']=roll_mean_f(cc,20)[off]/cur_c-1.0
        f['MA10']=roll_mean_f(cc,10)[off]/cur_c-1.0; f['STD20']=roll_std_f(ret,20)[off]
        f['MIN5']=af.roll_min_ignore(ll,5,2)[off]/cur_c-1.0; f['MIN20']=af.roll_min_ignore(ll,20,10)[off]/cur_c-1.0
        f['RET20']=cc[off-20]/cur_c-1.0; f['TURN5']=roll_mean_f(tt,5)[off]/100.0
        f['VSTD20']=roll_std_f(lv,20)[off]; f['AMT_RATIO60']=la[off]-roll_mean_f(la,60)[off]
        f['VMA60']=roll_mean_f(vv,60)[off]/(np.asarray(v[t],dtype=np.float64)+1e-12)-1.0
        dv=np.full_like(vv,np.nan); dv[1:]=np.diff(vv,axis=0)
        f['VSUMD60']=(roll_mean_f(np.where(np.isfinite(dv),np.maximum(dv,0),np.nan),60)[off]
                      -roll_mean_f(np.where(np.isfinite(dv),np.maximum(-dv,0),np.nan),60)[off])
        vr=np.full_like(vv,np.nan); vr[1:]=vv[1:]/vv[:-1]-1.0
        f['CORR20']=af.roll_corr(ret,np.log1p(np.abs(vr)),20,7)[off]
        vol60=roll_std_f(ret,60)[off]
        X=np.column_stack([np.ones(ns),f['LN_CAP'],vol60,f['TURN5']])
        def resid(y):
            out=np.full(ns,np.nan); m=np.isfinite(y)&np.all(np.isfinite(X),axis=1)
            if m.sum()<200: return out
            b,*_=np.linalg.lstsq(X[m],y[m],rcond=None); out[m]=y[m]-X[m]@b; return out
        for src,dst in (('MA20','MA20_resid'),('MA60','MA60_resid'),('VSTD20','VSTD20_resid'),('AMT_RATIO60','AMT_RATIO60_resid')):
            f[dst]=resid(f[src])
        return f
    sid_hash=np.array([abs(hash(str(s))) for s in syms],dtype=np.int64)
    n_bf=len(FEATS)-len(FUND)
    Xl=[];yl=[];dl=[];sl=[]
    for t in range(70,nd-H-1,5):
        m=valid[t]&np.isfinite(lab[t]); idx=np.where(m)[0]
        if len(idx)<300: continue
        if len(idx)>800:
            km=(sid_hash[idx]%5)==0
            if km.sum()>=300: idx=idx[km]
        bf=base_feats(t)
        Mx=np.column_stack([bf[k][idx] for k in FEATS if k not in FUND]+[FUND[k][t][idx] for k in FUND])
        Mb=Mx[:,:n_bf]; Mf=Mx[:,n_bf:].copy()
        ok=np.all(np.isfinite(Mb),axis=1)
        for j in range(Mf.shape[1]):
            col=Mf[:,j]; med=np.nanmedian(col)
            col[~np.isfinite(col)]=med if np.isfinite(med) else 0.0
            Mf[:,j]=col
        ok&=np.isfinite(lab[t][idx])
        if ok.sum()<200: continue
        Mx2=np.column_stack([Mb[ok],Mf[ok]])
        Xl.append(cs_prep(Mx2)); yl.append(lab[t][idx][ok]); dl.append(np.full(int(ok.sum()),t)); sl.append(idx[ok])
        del bf,Mx,Mb,Mf,Mx2; gc.collect()
        if t%100==0: w(f't={t} rows={sum(len(x) for x in Xl)} {round(time.time()-t0)}s')
    X=np.vstack(Xl).astype(np.float32); y=np.concatenate(yl).astype(np.float32)
    d=np.concatenate(dl); sidx=np.concatenate(sl)
    w(f'design {X.shape} {round(time.time()-t0)}s')
    C='CACHE/'
    np.save(C+'v5_X.npy',X); np.save(C+'v5_y.npy',y); np.save(C+'v5_d.npy',d); np.save(C+'v5_s.npy',sidx)
    json.dump(FEATS, open(C+'v5_feats.json','w'))
    w('DONE '+str(round(time.time()-t0)))
except Exception:
    w(traceback.format_exc())
