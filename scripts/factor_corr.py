"""因子相关性矩阵 + 聚类（星图连线与「因子星簇」的数据基础）。

口径：对每个截面算因子值的 Spearman 秩相关，再对时间取平均。
只用同一批股票、同一批日期，避免样本错配。
"""
import sys, json, time; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
import stellarq.factors.alpha_fast as af
OUT = DATA
R='WIDE/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
o,h,l,c,v,amt,turn,cap = M('open'),M('high'),M('low'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap')
valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
nd,ns=c.shape
def col(f,i0,i1): return np.asarray(f[i0:i1],dtype=np.float64)
FACTORS=['VSTD20','VSTD10','VSTD60','AMT_MA5','AMT_RATIO60','AMT_RATIO20','VSUMD60','VMA60','MA60','MA20','MA10',
         'CORR20','MIN5','MIN20','RET20','STD20','TURN5','KMID','LN_CAP','CLOSE_POS','VWAP_GAP']
def compute(name, i0, i1):
    import re
    m=re.match(r'([A-Z_]+?)(\d+)$', name); base,ww=(m.group(1),int(m.group(2))) if m else (name,0)
    s=max(0,i0-max(ww,20)); off=i0-s
    cc=col(c,s,i1)
    cur=col(c,i0,i1)
    if name=='KMID': return (cur-col(o,i0,i1))/col(o,i0,i1)
    if name=='LN_CAP': return np.log1p(col(cap,i0,i1))
    if name=='CLOSE_POS': return (col(c,i0,i1)-col(l,i0,i1))/(col(h,i0,i1)-col(l,i0,i1)+1e-12)
    if name=='VWAP_GAP': return col(c,i0,i1)/col(M('avg_price' if False else 'close'),i0,i1)-1.0 if False else col(c,i0,i1)/col(c,i0,i1)-np.nan
    if base=='MA': return af.roll_mean(cc,ww,max(2,ww//2))[off:]/cur-1.0
    if base=='STD':
        r=np.full_like(cc,np.nan); r[1:]=cc[1:]/cc[:-1]-1.0
        return af.roll_std(r,ww,max(2,ww//2))[off:]
    if base=='MIN': return af.roll_min_ignore(col(l,s,i1),ww,max(2,ww//2))[off:]/cur-1.0
    if base=='RET': return col(c,i0-ww,i1-ww)/cur-1.0
    if base=='TURN': return af.roll_mean(col(turn,s,i1),ww,max(2,ww//2))[off:]/100.0
    if base=='VMA': return af.roll_mean(col(v,s,i1),ww,max(2,ww//2))[off:]/(col(v,i0,i1)+1e-12)-1.0
    if base=='AMT_MA': return af.roll_mean(np.log1p(col(amt,s,i1)),ww,max(2,ww//2))[off:]
    if base=='AMT_RATIO':
        ext=np.log1p(col(amt,s,i1)); return ext[off:]-af.roll_mean(ext,ww,max(2,ww//2))[off:]
    if base=='VSTD': return af.roll_std(np.log1p(col(v,s,i1)),ww,max(2,ww//2))[off:]
    if base=='VSUMD':
        vv=col(v,s,i1); dv=np.full_like(vv,np.nan); dv[1:]=np.diff(vv,axis=0)
        pos=af.roll_mean(np.where(np.isfinite(dv),np.maximum(dv,0),np.nan),ww,max(2,ww//2))[off:]
        neg=af.roll_mean(np.where(np.isfinite(dv),np.maximum(-dv,0),np.nan),ww,max(2,ww//2))[off:]
        return (pos-neg)/(pos+neg+1e-12)
    if base=='CORR':
        r=np.full_like(cc,np.nan); r[1:]=cc[1:]/cc[:-1]-1.0
        vv=col(v,s,i1); vr=np.full_like(vv,np.nan); vr[1:]=vv[1:]/vv[:-1]-1.0
        return af.roll_corr(r,np.log1p(np.abs(vr)),ww,max(3,ww//3))[off:]
    raise KeyError(name)
FACTORS=[f for f in FACTORS if f!='VWAP_GAP']
n=len(FACTORS)
acc=np.zeros((n,n)); cnt=0
t0=time.time()
for t in range(300, nd, 40):
    i1=min(t+40, nd)
    vals=np.full((i1-t, n, ns), np.nan)
    for j,f in enumerate(FACTORS):
        vals[:,j,:]=compute(f,t,i1)
    for k in range(i1-t):
        m=valid[t+k]
        for j in range(n): vals[k,j]=(vals[k,j]-np.nanmean(vals[k,j][m]))/ (np.nanstd(vals[k,j][m])+1e-12)
    for k in range(i1-t):
        m=valid[t+k]&np.all(np.isfinite(vals[k]),axis=0)
        if m.sum()<300: continue
        blk=vals[k][:,m]
        rk=np.empty_like(blk)
        for j in range(n):
            rk[j]=np.argsort(np.argsort(blk[j],kind='mergesort'))
        rk-=rk.mean(axis=1,keepdims=True)
        sd=rk.std(axis=1); sd[sd==0]=1
        rk/=sd[:,None]
        acc+=rk@rk.T/(m.sum()); cnt+=1
    if t%800==0: print(f't={t} cs={cnt} {round(time.time()-t0)}s', flush=True)
corr=acc/max(cnt,1)
corr=np.clip(corr,-1,1)
df=pd.DataFrame(corr, index=FACTORS, columns=FACTORS)
df.to_csv(OUT / 'factor_corr.csv')
print('截面数', cnt)
# 简单聚类（贪心：按 |corr| 阈值分组）
import itertools
from stellarq.paths import DATA, ROOT, WIDE  # noqa: E402
groups=[]; used=set()
for j in range(n):
    if j in used: continue
    g=[j]; used.add(j)
    for k in range(j+1,n):
        if k not in used and abs(corr[j,k])>=0.7:
            g.append(k); used.add(k)
    groups.append({'cluster':len(groups),'factors':[FACTORS[i] for i in g]})
print(json.dumps(groups, ensure_ascii=False, indent=1))
json.dump({'factors':FACTORS,'corr':corr.tolist(),'clusters':groups,'n_cross_sections':cnt},
          open(OUT / 'factor_corr.json','w'), ensure_ascii=False)
