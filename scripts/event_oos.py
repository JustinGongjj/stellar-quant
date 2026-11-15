"""事件因子样本外确认：ev_abnormal_20d（异常波动公告）与 ev_reduce_60d（减持）在样本外是否成立。

同时检验一个更可操作的口径：「近 20 日内是否出现异常波动公告」的二元分组收益。
"""
import sys, glob, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.factors.event_fund import event_matrix
from stellarq.paths import CACHE, DATA, ROOT, WIDE  # noqa: E402
R='WIDE/'
o=np.load(R+'wide_open.npy', mmap_mode='r'); valid=np.load(R+'wide_valid.npy')
dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
nd,ns=o.shape; H=6
ev=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('CACHE/events/*.parquet'))], ignore_index=True)
ev=ev.drop_duplicates(subset=['symbol','ann_date','event'])
M=event_matrix(ev, dates, syms)
lab=np.full((nd,ns),np.nan,dtype=np.float32)
for i0 in range(0,nd-1-H,600):
    i1=min(i0+600,nd-1-H)
    lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
lab=np.where(valid,lab,np.nan)
cut=int(nd*0.85)
res={}
for cat in ('abnormal','reduce'):
    f=M[cat]
    for label,mask in (('train_all',np.arange(nd)<cut),('test_oos',np.arange(nd)>=cut)):
        ics=[]; grp=[]
        for t in range(70,nd-H-1):
            if not mask[t]: continue
            m=np.isfinite(lab[t])
            if m.sum()<500: continue
            hit=m&(f[t]>0); nohit=m&(f[t]==0)
            if hit.sum()<30 or nohit.sum()<200: continue
            a=lab[t][hit]; b=lab[t][nohit]
            # 事件组与对照组的平均收益差（原始，不做市值中性）
            grp.append(float(a.mean()-b.mean()))
            ra=np.argsort(np.argsort(f[t][m],kind='mergesort')).astype(float)
            rb=np.argsort(np.argsort(lab[t][m],kind='mergesort')).astype(float)
            if ra.std()>0 and rb.std()>0:
                ics.append(float(((ra-ra.mean())*(rb-rb.mean())).mean()/(ra.std()*rb.std())))
        if ics:
            arr=np.array(ics)
            res[f'{cat}_{label}']={'n_days':len(arr),'ic_mean':float(arr.mean()),
                                   'icir':float(arr.mean()/arr.std(ddof=1)),
                                   'excess_vs_nohit':float(np.mean(grp)),
                                   'excess_t':float(np.mean(grp)/(np.std(grp,ddof=1)/np.sqrt(len(grp)))) if np.std(grp,ddof=1)>0 else None}
            print(json.dumps(res[f'{cat}_{label}'], ensure_ascii=False), flush=True)
json.dump(res, open('DATA/event_oos.json','w'), ensure_ascii=False, indent=1)
