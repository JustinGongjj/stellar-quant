"""回测事件信号（pyarrow 分批版，峰值内存 <500MB）。"""
import numpy as np, pandas as pd, json, gc
import pyarrow.parquet as pq
from stellarq.paths import CACHE, DATA  # noqa: E402
C='CACHE/'
OUT = DATA
PATH=C+'a_share_daily_v2.parquet'
pf=pq.ParquetFile(PATH)
nrow=pf.metadata.num_rows
print('rows', nrow, 'rowgroups', pf.num_row_groups, flush=True)

# 按日期序号做基准累加（uint16 索引）与信号收集
bm1_sum={}; bm1_cnt={}; bm0_sum={}; bm0_cnt={}
sig_rows=[]
prev_tail=None      # 跨 batch 的尾部（同一股票需要连续）
cur_sym=None
carry=None

def process(blk):
    """blk 已按 symbol,date 排序，且同一 symbol 至少 30 行才处理。"""
    out_sig=[]
    for sym, g in blk.groupby('symbol', observed=True, sort=False):
        if len(g)<30: continue
        c=g['close'].to_numpy('float32'); h=g['high'].to_numpy('float32')
        l=g['low'].to_numpy('float32'); a=g['amount'].to_numpy('float32')
        d=g['date'].to_numpy()
        n=len(c)
        ret=np.empty(n,'float32'); ret[0]=np.nan; np.divide(c[1:],c[:-1],out=ret[1:]); ret[1:]-=1
        amt=pd.Series(a).rolling(20,min_periods=10).mean().to_numpy('float32')
        amp=np.empty(n,'float32'); amp[0]=np.nan
        np.divide(h[1:]-l[1:],c[:-1],out=amp[1:])
        f1=np.full(n,np.nan,'float32'); f0=np.full(n,np.nan,'float32')
        if n>6:
            np.divide(c[6:],c[1:n-5],out=f1[:n-6]); f1[:n-6]-=1
            np.divide(c[5:],c[:n-5],out=f0[:n-5]); f0[:n-5]-=1
        # 基准累加
        for i in np.where(np.isfinite(f1))[0]:
            k=int(d[i]); bm1_sum[k]=bm1_sum.get(k,0.0)+float(f1[i]); bm1_cnt[k]=bm1_cnt.get(k,0)+1
        for i in np.where(np.isfinite(f0))[0]:
            k=int(d[i]); bm0_sum[k]=bm0_sum.get(k,0.0)+float(f0[i]); bm0_cnt[k]=bm0_cnt.get(k,0)+1
        m=(ret>=0.07)&(a>=amt*2)&(amp>0.02)&np.isfinite(f1)&np.isfinite(f0)
        for i in np.where(m)[0]:
            out_sig.append((int(d[i]), float(f1[i]), float(f0[i])))
    return out_sig

for rg in range(pf.num_row_groups):
    t=pf.read_row_group(rg, columns=['date','symbol','close','high','low','amount'])
    blk=t.to_pandas()
    blk['date']=pd.to_datetime(blk['date']).astype('int64')//10**9//86400   # epoch day (int)
    blk=blk.sort_values(['symbol','date'], ignore_index=True)
    sig_rows.extend(process(blk))
    del t, blk; gc.collect()
    print(f'rg {rg+1}/{pf.num_row_groups} signals={len(sig_rows)}', flush=True)

print('total signals', len(sig_rows), flush=True)
S=pd.DataFrame(sig_rows, columns=['d','f1','f0'])
S['d']=S['d'].astype('int64')
b1=np.array([bm1_sum[k]/bm1_cnt[k] for k in sorted(bm1_sum)], 'float64')
b0=np.array([bm0_sum[k]/bm0_cnt[k] for k in sorted(bm0_sum)], 'float64')
k1=np.array(sorted(bm1_sum)); k0=np.array(sorted(bm0_sum))
m1=pd.Series(b1, index=k1); m0=pd.Series(b0, index=k0)
def stat(x,label):
    x=np.asarray(x,'float64'); x=x[np.isfinite(x)]
    if len(x)<10: return None
    return {'label':label,'n':int(len(x)),'mean':float(x.mean()),'median':float(np.median(x)),
            'win':float((x>0).mean()),'std':float(x.std(ddof=1))}
res={'A_追事件_T1买T6卖':stat(S['f1'],'A 追事件'),
     'B_T日持有T5卖':stat(S['f0'],'B 利好出尽'),
     'A_超额':stat(S['f1']-S['d'].map(m1),'A 超额'),
     'B_超额':stat(S['f0']-S['d'].map(m0),'B 超额'),
     'bench':stat(b1,'全市场等权')}
print()
for k,v in res.items():
    if v: print(f'{v["label"]:14s} n={v["n"]:7d} 均值{v["mean"]*100:+6.2f}% 中位{v["median"]*100:+6.2f}% 胜率{v["win"]*100:5.1f}%')
json.dump(res, open(OUT / 'event_signal_backtest.json','w'), ensure_ascii=False, indent=1)
print('saved')
