import sys, time, glob, gc; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd
import stellarq.factors.alpha_fast as af
from stellarq.validation.fast import cs_prep
from stellarq.factors.event_fund import fundamental_panel, point_in_time
from stellarq.models.ensemble import DoubleEnsembleRegressor, DEConfig, platt_fit, platt_apply, brier, ece
from stellarq.models.gbdt import GBDTParams
from stellarq.portfolio.costs import CostModel, stamp_tax_rate
R=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/wide2/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
o,c,v,amt,turn,cap,l = M('open'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap'),M('low')
valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
nd,ns=c.shape; H=6
lab=np.full((nd,ns),np.nan,dtype=np.float32)
for i0 in range(0,nd-1-H,600):
    i1=min(i0+600,nd-1-H)
    lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
lab=np.where(valid,lab,np.nan)
fp=fundamental_panel(sorted(glob.glob(str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/fund/*.parquet')), pd.Timestamp(dates[-1]))
mat=point_in_time(fp, dates, syms, ['WEIGHTAVG_ROE','SJLTZ','YSTZ','XSMLL','BASIC_EPS','BPS','MGJYXJJE'])
del fp; gc.collect()
print('mat ok', flush=True)
close=np.asarray(c,dtype=np.float32); close_safe=np.where(close>0,close,np.nan)
FUND={'f_bp':mat['BPS']/close_safe,'f_ep':mat['BASIC_EPS']/close_safe,'f_growth':mat['SJLTZ'],'f_rev_grow':mat['YSTZ'],
      'f_cfo':mat['MGJYXJJE']/close_safe,'f_rev_acc':mat['YSTZ']-mat['SJLTZ'],'f_margin':mat['XSMLL'],'f_roe':mat['WEIGHTAVG_ROE']}
for k in list(FUND):
    FUND[k]=np.where(valid,FUND[k],np.nan).astype(np.float32)
print('FUND ok', flush=True)
del mat, close_safe; gc.collect()
print('gc ok', flush=True)
