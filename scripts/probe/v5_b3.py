import sys, time, glob, gc; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd
from stellarq.factors.event_fund import fundamental_panel, point_in_time
R=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/wide2/'
c=np.load(R+'wide_close.npy', mmap_mode='r')
valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
nd,ns=c.shape
fp=fundamental_panel(sorted(glob.glob(str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/fund/*.parquet')), pd.Timestamp(dates[-1]))
mat=point_in_time(fp, dates, syms, ['WEIGHTAVG_ROE','SJLTZ','YSTZ','XSMLL','BASIC_EPS','BPS','MGJYXJJE'])
del fp; gc.collect()
print('mat ok', flush=True)
close=np.asarray(c,dtype=np.float32)
print('close ok', flush=True)
cs=np.array(close, dtype=np.float32, copy=True); cs[cs<=0]=np.nan
print('cs ok', flush=True)
out=np.array(mat['BPS'], dtype=np.float32, copy=True)
print('out ok', flush=True)
out[~(cs>0)]=np.nan
print('mask ok', flush=True)
out/=cs
print('div ok', flush=True)
out[~valid]=np.nan
print('valid ok', flush=True)
