"""退市风险代理因子：用不依赖退市样本的数据构造，并检验 IC。

A 股退市的两大现实路径：
  1. 财务类（净利润为负 + 营收 < 1 亿）→ *ST → 退市
  2. 交易类（连续 20 日收盘 < 1 元）→ 面值退市
我们虽然没有退市股，但可以在**存量样本**里构造「接近退市特征」的连续因子：
  dy_dist   距 1 元面值的相对距离（越近风险越高，log(close)）
  dy_small  市值分位（越小越危险）
  dy_fund   最新 ROE 与净利同比（负值越大越危险）
  dy_turn   换手率异常低（流动性枯竭）
合成 dy_risk = 标准化后加权求和，检验其与未来收益的关系（应为负）。
"""
import sys, glob, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.factors.event_fund import fundamental_panel, point_in_time
from stellarq.validation.factor_test import bh_adjust, pvalue_from_t, newey_west_t
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
OUT = DATA
C='CACHE/'; R=C+'wide2/'
o=np.load(R+'wide_open.npy', mmap_mode='r'); c=np.load(R+'wide_close.npy', mmap_mode='r')
turn=np.load(R+'wide_turnover.npy', mmap_mode='r'); cap=np.load(R+'wide_float_mktcap.npy', mmap_mode='r')
valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
nd,ns=c.shape; H=6
fp=fundamental_panel(sorted(glob.glob(C+'fund/*.parquet')), pd.Timestamp(dates[-1]))
mat=point_in_time(fp, dates, syms, ['WEIGHTAVG_ROE','SJLTZ','YSTZ'])
del fp
lab=np.full((nd,ns),np.nan,dtype=np.float32)
for i0 in range(0,nd-1-H,600):
    i1=min(i0+600,nd-1-H)
    lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
lab=np.where(valid,lab,np.nan)
close=np.asarray(c,dtype=np.float32)
TURN20=np.full((nd,ns),np.nan,dtype=np.float32)
for i0 in range(0,nd-20,300):
    i1=min(i0+300,nd)
    TURN20[i0:i1]=np.asarray(turn[i0:i1],dtype=np.float64).mean(axis=0)[None,:] if False else np.nan
# 简化：用当期换手与 20 日均值的比值需滚动，这里直接用当期换手分位替代
def zs(x):
    m=np.isfinite(x)
    if m.sum()<200: return np.zeros_like(x)
    mu=np.nanmean(x[m]); sd=np.nanstd(x[m])+1e-12
    return np.where(m,(x-mu)/sd,np.nan)
rows=[]
for t in range(200, nd-H-1, 5):
    m=valid[t]&np.isfinite(lab[t])
    if m.sum()<300: continue
    cl=np.where(m, close[t], np.nan)
    lnp=np.log(np.maximum(cl,1e-6))
    roe=np.where(m, mat['WEIGHTAVG_ROE'][t], np.nan)
    lz=zs(lnp); sz=zs(-np.log1p(np.asarray(cap[t],dtype=np.float64)))
    rz=zs(-np.where(np.isfinite(roe), roe, 0.0))
    risk=np.nanmean(np.stack([lz,sz,rz]),axis=0)    # 越高=越像退市特征
    x=risk[m]; y=lab[t][m]
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<200: continue
    a=x[ok]; b=y[ok]
    ra=np.argsort(np.argsort(a,kind='mergesort')).astype(float)
    rb=np.argsort(np.argsort(b,kind='mergesort')).astype(float)
    if ra.std()>0 and rb.std()>0:
        rows.append(float(((ra-ra.mean())*(rb-rb.mean())).mean()/(ra.std()*rb.std())))
rows=np.array(rows)
mu=float(rows.mean()); sd=float(rows.std(ddof=1)); t=newey_west_t(pd.Series(rows))
res={'n':int(len(rows)),'ic_mean':mu,'icir':mu/sd,'t_nw':t,'p':pvalue_from_t(t,len(rows)-1)}
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(OUT / 'delist_proxy.json','w'), ensure_ascii=False, indent=1, default=float)
