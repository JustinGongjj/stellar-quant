import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.paths import CACHE, ROOT  # noqa: E402
C='CACHE/'
st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
R=C+'wide2/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
o=M('open'); valid=np.load(R+'wide_valid.npy'); syms=np.load(R+'wide_syms.npy',allow_pickle=True); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
dl=list(np.unique(dte))
print('dates', len(dl), dl[:3])
# 原始得分与平滑得分的类内方差
u=dl[5]
raw=pd.Series(st[dte==u], index=[str(syms[i]) for i in ste[dte==u]])
print('n prev1', int((dte==dl[4]).sum()), 'n prev2', int((dte==dl[3]).sum()), 'n now', int((dte==u).sum()))
p1=pd.Series(st[dte==dl[4]], index=[str(syms[i]) for i in ste[dte==dl[4]]])
p2=pd.Series(st[dte==dl[3]], index=[str(syms[i]) for i in ste[dte==dl[3]]])
print('overlap now&prev1', len(set(raw.index)&set(p1.index)), 'prev1&prev2', len(set(p1.index)&set(p2.index)))
sm=(raw.add(p1,fill_value=np.nan).add(p2,fill_value=np.nan))/3.0
print('raw std', float(raw.std()), 'smooth std', float(sm.std()), 'smooth n', int(sm.notna().sum()), 'raw n', len(raw))
print('top5 raw', list(raw.sort_values(ascending=False).head(5).index))
print('top5 smooth', list(sm.sort_values(ascending=False).head(5).index))
uu=int(u)
rp=pd.Series(np.asarray(o[uu+7],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0, index=[str(s) for s in syms])
for nm,sc in (('raw',raw),('smooth',sm)):
    top=list(sc.sort_values(ascending=False).head(50).index)
    r=rp.reindex(top).dropna()
    print(nm,'top50 mean', round(float(r.mean()),5), 'n', len(r))
