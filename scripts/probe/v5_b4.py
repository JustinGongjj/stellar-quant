import sys, glob, gc, resource; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd
from stellarq.factors.event_fund import fundamental_panel, point_in_time
def mem(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
R=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/wide2/'
c=np.load(R+'wide_close.npy', mmap_mode='r'); valid=np.load(R+'wide_valid.npy')
dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
fp=fundamental_panel(sorted(glob.glob(str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/fund/*.parquet')), pd.Timestamp(dates[-1]))
mat=point_in_time(fp, dates, syms, ['WEIGHTAVG_ROE','SJLTZ','YSTZ','XSMLL','BASIC_EPS','BPS','MGJYXJJE'])
del fp; gc.collect()
print('after pit', mem(), flush=True)
close=np.asarray(c,dtype=np.float32); cs=np.array(close,dtype=np.float32,copy=True); cs[cs<=0]=np.nan
F={}
for name,num,den in [('bp',mat['BPS'],cs),('ep',mat['BASIC_EPS'],cs),('growth',mat['SJLTZ'],None),('rev',mat['YSTZ'],None),('cfo',mat['MGJYXJJE'],cs),('margin',mat['XSMLL'],None),('roe',mat['WEIGHTAVG_ROE'],None)]:
    out=np.array(num,dtype=np.float32,copy=True)
    if den is not None: out/=den
    out[~valid]=np.nan
    F[name]=out
    print(name, mem(), flush=True)
