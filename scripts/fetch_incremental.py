"""增量抓取：补齐 2026-10-01 之后的最新日线（腾讯接口已恢复）。

只抓「面板里已存在的股票」（5,027 只），不做全市场重抓，省时间也避开限流。
每只只取最近 200 根，与面板日期对齐后判断是否需要更新。
"""
import sys, time, json, os; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from stellarq.data.tencent import Tencent, to_tx_symbol
from stellarq.paths import DATA, ROOT  # noqa: E402
OUT = DATA
CACHE = str(__import__('pathlib').Path.home() / '.cache/stellarq')
def w(m): print(m, flush=True)
R=CACHE+'/wide2/'
syms=[str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)]
w(f'target symbols {len(syms)}')
cli=Tencent(pause=0.0, timeout=20)
def one(sym):
    try:
        d=cli.daily(sym)
        if d.empty: return sym, None
        d=d[d['date']>=pd.Timestamp('2026-09-25')]
        return sym, d
    except Exception as e:
        return sym, None
t0=time.time(); rows=[]; fails=0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs={ex.submit(one,s):s for s in syms}
    done=0
    for f in as_completed(futs):
        sym,d=f.result(); done+=1
        if d is None or d.empty:
            fails+=1
        else:
            d=d.copy(); d['symbol']=sym
            rows.append(d[['symbol','date','open','high','low','close','volume']])
        if done%1000==0: w(f'{done}/{len(syms)} ok={len(rows)} fail={fails} {round(time.time()-t0)}s')
if rows:
    df=pd.concat(rows, ignore_index=True)
    df.to_parquet(CACHE+'/incremental_daily.parquet', index=False)
    w(f'saved {len(df)} rows, {df["date"].min().date()} -> {df["date"].max().date()}, symbols {df["symbol"].nunique()}, fails {fails}')
else:
    w('no rows')
