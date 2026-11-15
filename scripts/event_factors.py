"""事件因子检验：把巨潮公告做成「过去 N 个交易日的事件计数」，检验 IC。
from stellarq.paths import CACHE, DATA, ROOT, WIDE  # noqa: E402

事件类别（EVENT_WINDOW 决定窗口）：
  业绩预告 20 日、业绩快报 20 日、回购 60 日、增持 60 日、减持 60 日、
  重组 120 日、处罚 120 日、股权激励 120 日、异常波动 20 日。
"""
import sys, glob, json, time, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    from stellarq.factors.event_fund import event_matrix, EVENT_COLS, EVENT_WINDOW
    from stellarq.validation.factor_test import bh_adjust, pvalue_from_t, newey_west_t
    t0=time.time()
    files=sorted(glob.glob('CACHE/events/*.parquet'))
    ev=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ev=ev.drop_duplicates(subset=['symbol','ann_date','event'])
    w(f'events {len(ev)} from {len(files)} files')
    w('by event: '+json.dumps(ev['event'].value_counts().to_dict(), ensure_ascii=False))
    R='WIDE/'
    o=np.load(R+'wide_open.npy', mmap_mode='r'); valid=np.load(R+'wide_valid.npy')
    dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
    nd,ns=o.shape; H=6
    Mdf=event_matrix(ev, dates, syms); w('matrix done')
    lab=np.full((nd,ns),np.nan,dtype=np.float32)
    for i0 in range(0,nd-1-H,600):
        i1=min(i0+600,nd-1-H)
        lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
    lab=np.where(valid,lab,np.nan)
    rows=[]
    WMAP={'forecast':'forecast','express':'express','buyback':'buyback','increase':'increase','reduce':'reduce',
          'restructure':'restructure','penalty':'penalty','equity_incentive':'equity_incentive','abnormal':'abnormal'}
    for col in EVENT_COLS:
        key=[k for k in WMAP if col.startswith('ev_'+k+'_')]
        if not key: continue
        cat=key[0]
        f=Mdf.pop(cat)
        hits=int((f>0).sum())
        if hits<1000:
            w(f'{col}: 事件过少 {hits}'); continue
        ics=[]
        for t in range(70, nd-H-1):
            m=(f[t]>0)&np.isfinite(lab[t])
            if m.sum()<60: continue        # 事件样本少，门槛放低
            a=f[t][m]; b=lab[t][m]
            if np.std(a)==0: continue
            ra=np.argsort(np.argsort(a,kind='mergesort')).astype(float)
            rb=np.argsort(np.argsort(b,kind='mergesort')).astype(float)
            if ra.std()>0 and rb.std()>0:
                ics.append(float(((ra-ra.mean())*(rb-rb.mean())).mean()/(ra.std()*rb.std())))
        if len(ics)<100:
            w(f'{col}: 截面不足 {len(ics)}'); continue
        arr=np.array(ics); mu=float(arr.mean()); sd=float(arr.std(ddof=1))
        # 事件组的平均超额（相对当日全样本中位）
        rows.append({'factor':col,'event_days':hits,'n_cs':len(arr),'ic_mean':mu,'icir':mu/sd if sd else np.nan,
                     't_nw':newey_west_t(pd.Series(arr))})
        w(f'{col:22s} 事件日 {hits:6d} IC {mu:+.4f} ICIR {rows[-1]["icir"]:+.3f} t {rows[-1]["t_nw"]:+.1f}')
    rep=pd.DataFrame(rows)
    if len(rep):
        rep['p_value']=[pvalue_from_t(t,max(int(n)-1,1)) for t,n in zip(rep['t_nw'],rep['n_cs'])]
        rep['q_bh']=bh_adjust(rep['p_value'].fillna(1).tolist())
        rep.to_csv(OUT / 'event_factors.csv',index=False)
        w(rep.round(4).to_string())
    w('DONE '+str(round(time.time()-t0)))
except Exception:
    w(traceback.format_exc())
