import sys, glob, gc, resource; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd
import stellarq.factors.alpha_fast as af
def mem(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
R=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/wide2/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
o,c,v,amt,turn,cap,l = M('open'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap'),M('low')
valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
nd,ns=c.shape
print('arrays', mem(), flush=True)
t=100
s=max(0,t-65); off=t-s
cc=np.asarray(c[s:t+1],dtype=np.float64)
vv=np.asarray(v[s:t+1],dtype=np.float64); aa=np.asarray(amt[s:t+1],dtype=np.float64)
print('cc/vv', mem(), flush=True)
lv=np.log1p(vv); la=np.log1p(aa)
print('log', mem(), flush=True)
m=af.roll_mean(lv,60,30)
print('roll_mean', mem(), flush=True)
st=af.roll_std(lv,20,10)
print('roll_std', mem(), flush=True)
mn=af.roll_min_ignore(np.asarray(l[s:t+1],dtype=np.float64),5,2)
print('roll_min', mem(), flush=True)
cor=af.roll_corr(np.diff(cc,axis=0),np.diff(vv,axis=0),20,7)
print('roll_corr', mem(), flush=True)
