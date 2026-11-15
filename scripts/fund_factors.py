"""基本面因子：点对点（point-in-time）版本，检验 IC。
from stellarq.paths import CACHE, DATA, ROOT, WIDE  # noqa: E402

构造（全部用公告日对齐，避免前视）：
  f_roe      ROE（近 4 季）
  f_growth   单季归母净利同比增速 SJLTZ
  f_rev_grow 单季营收同比 YSTZ
  f_margin   毛利率 XSMLL
  f_eps_ttm  滚动 4 季 EPS
  f_bp       BPS / 收盘价（账面市值比）
  f_ep       EPS_ttm / 收盘价
  f_quality  经营现金流/营收（MGJYXJJE/EPS 近似，或直接用 MGJYXJJE 标准化）
  f_rev_acc  营收同比 − 净利同比（增长质量：利润是否跟得上收入）
"""
import sys, glob, json, traceback, time; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'fund_factors.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'fund_factors.log','w').close()
try:
    import numpy as np, pandas as pd
    from stellarq.factors.event_fund import fundamental_panel, point_in_time
    from stellarq.validation.factor_test import bh_adjust, pvalue_from_t, newey_west_t
    t0=time.time()
    R='WIDE/'
    c=np.load(R+'wide_close.npy', mmap_mode='r'); o=np.load(R+'wide_open.npy', mmap_mode='r')
    cap=np.load(R+'wide_float_mktcap.npy', mmap_mode='r')
    valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
    nd,ns=c.shape
    files=sorted(glob.glob('CACHE/fund/*.parquet'))
    fp=fundamental_panel(files, pd.Timestamp(dates[-1]))
    w(f'fundamental rows {len(fp)} symbols {fp["symbol"].nunique()} {round(time.time()-t0)}s')
    FIELDS=['WEIGHTAVG_ROE','SJLTZ','YSTZ','XSMLL','BASIC_EPS','BPS','MGJYXJJE']
    mat=point_in_time(fp, dates, syms, FIELDS)
    w(f'pit done {round(time.time()-t0)}s')
    close=np.asarray(c,dtype=np.float32); capv=np.asarray(cap,dtype=np.float32)
    # TTM EPS：近似用最近一条 BASIC_EPS × 4（季度累积口径因季度而异，这里做稳健处理：用最新季度 EPS 年化）
    eps=mat['BASIC_EPS']; bps=mat['BPS']
    feats={
      'f_roe':mat['WEIGHTAVG_ROE'],
      'f_growth':mat['SJLTZ'],
      'f_rev_grow':mat['YSTZ'],
      'f_margin':mat['XSMLL'],
      'f_bp':bps/np.where(close>0,close,np.nan),
      'f_ep':eps/np.where(close>0,close,np.nan),
      'f_cfo':mat['MGJYXJJE']/np.where(close>0,close,np.nan),
      'f_rev_acc':mat['YSTZ']-mat['SJLTZ'],
      'f_roe_gap':mat['WEIGHTAVG_ROE']-np.nanmean(np.where(np.isfinite(mat['WEIGHTAVG_ROE']),mat['WEIGHTAVG_ROE'],np.nan),axis=1,keepdims=True),
    }
    # 标签：T+1 开盘买、T+6 开盘卖
    H=6
    lab=np.full((nd,ns),np.nan,dtype=np.float32)
    for i0 in range(0,nd-1-H,600):
        i1=min(i0+600,nd-1-H)
        lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
    lab=np.where(valid,lab,np.nan)
    rows=[]
    for name,f in feats.items():
        f=np.where(valid,f,np.nan).astype(np.float64)
        ics=[]
        for t in range(70, nd-H-1):
            m=np.isfinite(f[t])&np.isfinite(lab[t])
            if m.sum()<200: continue
            a=f[t][m]; b=lab[t][m]
            ra=np.argsort(np.argsort(a,kind='mergesort')).astype(float)
            rb=np.argsort(np.argsort(b,kind='mergesort')).astype(float)
            sa,sb=ra.std(),rb.std()
            if sa>0 and sb>0: ics.append(float(((ra-ra.mean())*(rb-rb.mean())).mean()/(sa*sb)))
        if len(ics)<200: 
            w(f'{name}: 样本不足 {len(ics)}'); continue
        arr=np.array(ics); mu=float(arr.mean()); sd=float(arr.std(ddof=1))
        rows.append({'factor':name,'n':len(arr),'ic_mean':mu,'icir':mu/sd if sd else np.nan,
                     't_nw':newey_west_t(pd.Series(arr)),'covered_days':int(len(arr))})
        w(f'{name:12s} IC {mu:+.4f} ICIR {rows[-1]["icir"]:+.3f} t {rows[-1]["t_nw"]:+.1f} n {len(arr)} {round(time.time()-t0)}s')
    rep=pd.DataFrame(rows)
    rep['p_value']=[pvalue_from_t(t,max(int(n)-1,1)) for t,n in zip(rep['t_nw'],rep['n'])]
    rep['q_bh']=bh_adjust(rep['p_value'].fillna(1).tolist())
    rep.to_csv(OUT / 'fund_factors.csv',index=False)
    w(rep.round(4).to_string())
    w('DONE '+str(round(time.time()-t0)))
except Exception:
    w(traceback.format_exc())
