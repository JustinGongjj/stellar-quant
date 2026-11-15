"""EM 不复权收盘价的涨跌停合规性检验（数据质量的核心证据）。

方法：对每只股票，用「前一日收盘」与「当日收盘」算收益，统计超过板块涨跌停上限的天数占比。
若 EM 收盘价可用，主板应几乎全部 ≤10%（留 0.5 个百分点的容差），创业板/科创板 ≤20%。
"""
import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd, pyarrow.parquet as pq
F=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/a_share_daily_v2.parquet'
cols=['symbol','date','close','name']
df=pq.read_table(F, columns=cols).to_pandas()
df['date']=pd.to_datetime(df['date'])
df=df.sort_values(['symbol','date'])
df['ret']=df.groupby('symbol')['close'].pct_change()
board=np.where(df['symbol'].str.startswith(('688','689')),20.0,
        np.where(df['symbol'].str.startswith(('300','301')),20.0,10.0))
df['limit']=board
df['excess']=df['ret'].abs()*100 - df['limit']
g=df.groupby('symbol').agg(n=('ret','count'), bad=('excess', lambda x: int((x>0.6).sum())))
g=g[g['n']>200]
print('股票数', len(g), '总交易日', int(g['n'].sum()), '超涨跌停天数', int(g['bad'].sum()),
      '占比', round(g['bad'].sum()/g['n'].sum()*100,4), '%')
print('有超限的股票占比', round((g['bad']>0).mean()*100,2), '%')
print(g.sort_values('bad',ascending=False).head(5))
