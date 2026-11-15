"""回测 v2：加入对照组与分层，回答三个问题。

Q1 「追事件」真的亏吗？→ 与同期全市场比（超额）
Q2 「利好出尽」是策略吗？→ 两组是同一批股票的两个时点，不能直接比
Q3 有没有可交易的变体？→ 按涨幅档位、按市值分层
"""
import numpy as np, pandas as pd, json, gc
import pyarrow.parquet as pq
from stellarq.paths import CACHE, DATA  # noqa: E402
C='CACHE/'
OUT = DATA
pf=pq.ParquetFile(C+'a_share_daily_v2.parquet')
bm1_s={};bm1_c={};bm0_s={};bm0_c={}
sig=[]
def process(blk):
    for sym,g in blk.groupby('symbol',observed=True,sort=False):
        if len(g)<40: continue
        c=g['close'].to_numpy('float32');h=g['high'].to_numpy('float32')
        l=g['low'].to_numpy('float32');a=g['amount'].to_numpy('float32')
        d=g['date'].to_numpy()
        n=len(c)
        ret=np.empty(n,'float32');ret[0]=np.nan;np.divide(c[1:],c[:-1],out=ret[1:]);ret[1:]-=1
        amt=pd.Series(a).rolling(20,min_periods=10).mean().to_numpy('float32')
        amp=np.empty(n,'float32');amp[0]=np.nan;np.divide(h[1:]-l[1:],c[:-1],out=amp[1:])
        f1=np.full(n,np.nan,'float32');f0=np.full(n,np.nan,'float32')
        f5=np.full(n,np.nan,'float32')   # T+1 买 T+6 卖
        if n>6:
            np.divide(c[6:],c[1:n-5],out=f1[:n-6]);f1[:n-6]-=1
            np.divide(c[5:],c[:n-5],out=f0[:n-5]);f0[:n-5]-=1
        for i in np.where(np.isfinite(f0))[0]:
            k=int(d[i]);bm0_s[k]=bm0_s.get(k,0.)+float(f0[i]);bm0_c[k]=bm0_c.get(k,0)+1
        m=(ret>=0.07)&(a>=amt*2)&(amp>0.02)&np.isfinite(f1)&np.isfinite(f0)
        for i in np.where(m)[0]:
            sig.append((int(d[i]),float(f0[i]),float(f1[i]),float(ret[i]),float(a[i])))
for rg in range(pf.num_row_groups):
    t=pf.read_row_group(rg,columns=['date','symbol','close','high','low','amount'])
    blk=t.to_pandas()
    blk['date']=pd.to_datetime(blk['date']).astype('int64')//86400//10**9
    blk=blk.sort_values(['symbol','date'],ignore_index=True)
    process(blk); del t,blk; gc.collect()
    print('rg',rg+1,'sig',len(sig),flush=True)

S=pd.DataFrame(sig,columns=['d','f0','f1','ret','amt'])
b0=pd.Series({k:bm0_s[k]/bm0_c[k] for k in bm0_s})
S['ex0']=S['f0']-S['d'].map(b0)
def stat(x,l):
    x=np.asarray(x,'float64');x=x[np.isfinite(x)]
    if len(x)<20: return None
    return {'label':l,'n':int(len(x)),'mean':float(x.mean()),'win':float((x>0).mean()),
            't':float(x.mean()/(x.std(ddof=1)/np.sqrt(len(x))))}
res={}
print()
res['B_持有T5超额']=stat(S['ex0'],'B T日持有5日超额')
res['A_追高T1买超额']=stat(S['f1']-S['d'].map(b0),'A T+1买入超额')
# 分层：按当日涨幅
for lo,hi,l in [(0.07,0.09,'涨幅7-9%'),(0.09,0.15,'9-15%'),(0.15,9,'>15%(涨停)')]:
    sub=S[(S['ret']>=lo)&(S['ret']<hi)]
    res[f'B_{l}']=stat(sub['ex0'],f'B {l}')
    res[f'A_{l}']=stat(sub['f1']-sub['d'].map(b0),f'A {l}')
# 年化分年
S['yr']=pd.to_datetime(S['d']*86400,unit='s').dt.year
for y,g in S.groupby('yr'):
    if len(g)>500: res[f'B_{y}']=stat(g['ex0'],f'B {y}年')
for k,v in res.items():
    if v: print(f'{v["label"]:20s} n={v["n"]:7d} 超额{v["mean"]*100:+6.2f}% 胜率{v["win"]*100:5.1f}% t={v["t"]:+5.2f}')
json.dump(res,open(OUT / 'event_signal_backtest2.json','w'),ensure_ascii=False,indent=1)
print('saved')
