"""模型 v5（复用 v4 已缓存的量价设计矩阵，只把基本面因子追加进去）。

v4 的设计矩阵已经落盘（v4_X/y/d/s），但缺基本面列，且 v4 的抽样规则是「按股票固定」。
这里用同一套 sid_hash 规则重建「时刻 → 股票」的对应，把基本面按 (t, symbol) 对齐后拼上去。
"""
import sys, glob, gc, json, time, resource; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
def mem(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
def w(m): print(f'{m} [mem {mem()}MB]', flush=True)
w('boot')
import numpy as np, pandas as pd
from stellarq.factors.event_fund import fundamental_panel, point_in_time
from stellarq.paths import CACHE, ROOT, WIDE  # noqa: E402
R='WIDE/'; C='CACHE/'
X=np.load(C+'v4_X.npy'); y=np.load(C+'v4_y.npy'); d=np.load(C+'v4_d.npy'); sidx=np.load(C+'v4_s.npy')
w(f'v4 design {X.shape}')
dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
valid=np.load(R+'wide_valid.npy', mmap_mode='r')
fp=fundamental_panel(sorted(glob.glob(C+'fund/*.parquet')), pd.Timestamp(dates[-1]))
w(f'fp {fp.shape}')
mat=point_in_time(fp, dates, syms, ['SJLTZ','YSTZ','XSMLL','BASIC_EPS','BPS','MGJYXJJE','WEIGHTAVG_ROE'])
del fp; gc.collect()
w('pit done')
FEATS_F=['f_bp','f_ep','f_growth','f_rev_grow','f_cfo','f_rev_acc','f_margin','f_roe']
extra=np.empty((len(d), len(FEATS_F)), dtype=np.float32)
close=np.load(R+'wide_close.npy', mmap_mode='r')
for i,(fs,si) in enumerate(zip(d, sidx)):
    pass
w('alloc extra')
t_arr=d.astype(np.int64); s_arr=sidx.astype(np.int64)
close_v=np.asarray(close[t_arr, s_arr], dtype=np.float64)
eps=np.asarray(mat['BASIC_EPS'][t_arr, s_arr], dtype=np.float64)
bps=np.asarray(mat['BPS'][t_arr, s_arr], dtype=np.float64)
cfo=np.asarray(mat['MGJYXJJE'][t_arr, s_arr], dtype=np.float64)
growth=np.asarray(mat['SJLTZ'][t_arr, s_arr], dtype=np.float32)
rev=np.asarray(mat['YSTZ'][t_arr, s_arr], dtype=np.float32)
margin=np.asarray(mat['XSMLL'][t_arr, s_arr], dtype=np.float32)
roe=np.asarray(mat['WEIGHTAVG_ROE'][t_arr, s_arr], dtype=np.float32)
del mat; gc.collect()
w('gathered')
cs=np.where(close_v>0, close_v, np.nan)
extra[:,0]=bps/cs; extra[:,1]=eps/cs; extra[:,2]=growth; extra[:,3]=rev
extra[:,4]=cfo/cs; extra[:,5]=rev-growth; extra[:,6]=margin; extra[:,7]=roe
extra[~np.isfinite(extra)]=np.nan
np.save(C+'v5_extra.npy', extra)
w(f'extra saved {extra.shape} nan_ratio {np.isnan(extra).mean():.3f}')
