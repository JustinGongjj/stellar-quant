"""打板研究（内存安全版）：逐块统计，不整块 float64 化。
from stellarq.paths import DATA, ROOT, WIDE  # noqa: E402

日线能回答：涨停/炸板频次、连板高度、次日开盘溢价、高位连板衰减。
日线不能回答：能否买到（封单排队）、盘中开板、席位。故结论是**上界**。
"""
import sys, time, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'limitup.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'limitup.log','w').close()
try:
    import numpy as np, pandas as pd
    t0=time.time()
    R='WIDE/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o,h,c,amt = M('open'),M('high'),M('close'),M('amount')
    valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
    nd,ns=c.shape
    lim=np.array([0.20 if str(s).startswith(('688','689','300','301')) else 0.10 for s in syms],dtype=np.float64)
    n_zt=0; n_zb=0
    buckets={}   # 连板高度 -> 次日溢价列表
    first_board=[]; zb_next=[]
    by_year=[]
    cons=np.zeros(ns,dtype=np.int16)
    for t in range(1,nd-1):
        prev=np.asarray(c[t-1],dtype=np.float64)
        cur=np.asarray(c[t],dtype=np.float64)
        hi=np.asarray(h[t],dtype=np.float64)
        nxt_o=np.asarray(o[t+1],dtype=np.float64)
        ok=valid[t]&valid[t+1]&np.isfinite(prev)&(prev>0)
        lpx=np.round(prev*(1+lim)+1e-9,2)
        zt=ok&(cur>=lpx-1e-9)
        zb=ok&(hi>=lpx-1e-9)&(cur<lpx-1e-9)
        cons=np.where(zt, cons+1, 0)
        r=np.where(ok&(cur>0), nxt_o/cur-1.0, np.nan)
        n_zt+=int(zt.sum()); n_zb+=int(zb.sum())
        for k in (1,2,3,4):
            m=zt&(cons==k)&np.isfinite(r)
            if m.any(): buckets.setdefault(k,[]).append(r[m])
        m5=zt&(cons>=5)&np.isfinite(r)
        if m5.any(): buckets.setdefault(5,[]).append(r[m5])
        m1=zt&(cons==1)&np.isfinite(r)
        if m1.any(): first_board.append(r[m1])
        mz=zb&np.isfinite(r)
        if mz.any(): zb_next.append(r[mz])
        if t%400==0: w(f't={t} {round(time.time()-t0)}s')
    def stat(arrs, label):
        if not arrs: return None
        v=np.concatenate(arrs)
        return {'label':label,'n':int(len(v)),'mean':float(v.mean()),'median':float(np.median(v)),
                'win':float((v>0).mean()),'p10':float(np.percentile(v,10)),'p90':float(np.percentile(v,90))}
    res={'n_zt':n_zt,'n_zb':n_zb,'zb_rate':n_zb/max(n_zt+n_zb,1),
         'boards':[stat(buckets.get(k), f'{k}板') for k in (1,2,3,4,5) if buckets.get(k)],
         'failed':stat(zb_next,'炸板')}
    json.dump(res, open(OUT / 'limitup.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    w('DONE '+str(round(time.time()-t0)))
except Exception:
    w(traceback.format_exc())
