import sys, glob; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd, time
from stellarq.factors.event_fund import fundamental_panel, point_in_time
t0=time.time()
R=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/wide2/'
dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
fp=fundamental_panel(sorted(glob.glob(str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/fund/*.parquet')), pd.Timestamp(dates[-1]))
m=point_in_time(fp, dates[:400], syms, ['WEIGHTAVG_ROE','BPS'])
print('shape', m['BPS'].shape, 'time', round(time.time()-t0,1))
print('non-nan', int(np.isfinite(m['BPS']).sum()))
