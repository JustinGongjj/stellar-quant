"""多头端敏感性：冲击成本倍率、持有期、持仓数对结果的影响。"""
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
import sys, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'long_only_sens.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'long_only_sens.log','w').close()
try:
    import numpy as np, pandas as pd
    C='CACHE/'
    st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
    R=C+'wide2/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); amt=M('amount'); valid=np.load(R+'wide_valid.npy')
    syms=np.load(R+'wide_syms.npy',allow_pickle=True); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    H=6
    COMM=0.00025; TRANS=0.00001; SLIP=0.0005
    def stamp(dt): return 0.0005 if pd.Timestamp(dt)>=pd.Timestamp('2023-08-28') else 0.001
    dl=list(np.unique(dte))
    sc_by_date={u: pd.Series(st[dte==u], index=[str(syms[i]) for i in ste[dte==u]]) for u in dl}
    def run(topk=50, buf_in=60, buf_out=80, mult=1.0, size=1e8):
        prev=set(); per=[]
        for u in dl:
            uu=int(u); sc=sc_by_date[u]
            ranked=sc.sort_values(ascending=False); rank={s:i for i,s in enumerate(ranked.index)}
            keep=[s for s in prev if rank.get(s,10**9)<buf_out]
            held=list(dict.fromkeys(keep+list(ranked.head(buf_in).index)))[:topk]
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=[str(s) for s in syms])
            r=rp.reindex(held).dropna()
            if len(r)<topk*0.8: continue
            adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=[str(s) for s in syms]).reindex(held)
            pos=size/len(held)
            impact=float(np.nanmean(np.minimum(0.003, pos/np.maximum(adv.values,1e4)*0.1)))*mult
            new=set(held); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
            cost=turn_*(COMM+TRANS+SLIP+impact)+turn_*0.5*stamp(dates[uu])
            per.append({'gross':float(r.mean()),'net':float(r.mean())-cost,'bench':float(rp.dropna().mean())})
        pf=pd.DataFrame(per)
        ov=pf['net'].rolling(5).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(5).mean().dropna()
        return {'ann_net':float(ov.mean()*10.08),'ann_excess':float(ex.mean()*10.08),
                'sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
                'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),
                'cost':float((pf['gross']-pf['net']).mean())}
    res={}
    for mult in (0.0,1.0,2.0,4.0):
        res[f'impact_x{mult}']=run(mult=mult)
    for k in (20,50,100):
        res[f'top{k}']=run(topk=k, buf_in=int(k*1.2), buf_out=int(k*1.6))
    for size in (1e7,1e8,1e9):
        res[f'size_{int(size/1e6)}M']=run(size=size)
    json.dump(res, open(OUT / 'long_only_sens.json','w'), ensure_ascii=False, indent=1)
    w(json.dumps(res, ensure_ascii=False, indent=1))
except Exception:
    w(traceback.format_exc())
