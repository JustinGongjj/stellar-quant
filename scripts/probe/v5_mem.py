import sys, glob, resource; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd
def mem(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
print('start', mem(), flush=True)
R=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/wide2/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
o,c,v,amt,turn,cap,l = M('open'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap'),M('low')
print('mmap', mem(), flush=True)
valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
print('axes', mem(), flush=True)
from stellarq.factors.event_fund import fundamental_panel, point_in_time
fp=fundamental_panel(sorted(glob.glob(str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/fund/*.parquet')), pd.Timestamp(dates[-1]))
print('fp', mem(), flush=True)
mat=point_in_time(fp, dates, syms, ['WEIGHTAVG_ROE','SJLTZ','YSTZ','XSMLL','BASIC_EPS','BPS','MGJYXJJE'])
print('pit', mem(), flush=True)
