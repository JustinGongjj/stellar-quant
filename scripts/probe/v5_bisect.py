import sys, time, glob, gc; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd
print('A', flush=True)
OUT=str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'reports/data') + '/'
import stellarq.factors.alpha_fast as af
from stellarq.validation.fast import cs_prep
from stellarq.factors.event_fund import fundamental_panel, point_in_time
from stellarq.models.ensemble import DoubleEnsembleRegressor, DEConfig, platt_fit, platt_apply, brier, ece
from stellarq.models.gbdt import GBDTParams
from stellarq.portfolio.costs import CostModel, stamp_tax_rate
print('B imports ok', flush=True)
R=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/wide2/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
o,c,v,amt,turn,cap,l = M('open'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap'),M('low')
valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
nd,ns=c.shape; H=6
print('C arrays', nd, ns, flush=True)
lab=np.full((nd,ns),np.nan,dtype=np.float32)
for i0 in range(0,nd-1-H,600):
    i1=min(i0+600,nd-1-H)
    lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
lab=np.where(valid,lab,np.nan)
print('D label', flush=True)
files=sorted(glob.glob(str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/fund/*.parquet'))
fp=fundamental_panel(files, pd.Timestamp(dates[-1]))
print('E fp', fp.shape, flush=True)
mat=point_in_time(fp, dates, syms, ['WEIGHTAVG_ROE','SJLTZ','YSTZ','XSMLL','BASIC_EPS','BPS','MGJYXJJE'])
print('F pit', flush=True)
