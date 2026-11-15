"""CTA 数据抓取：18 个商品/股指连续合约。"""
import sys, json, time, os; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from stellarq.data.sina_futures import SinaFutures, UNIVERSE
from stellarq.paths import CACHE as stellarq_cache,  CACHE, DATA, ROOT  # noqa: E402
OUT = DATA
CACHE = stellarq_cache / 'futures.parquet'
cli=SinaFutures(pause=0.3)
frames=[]
for s,(name,sector) in UNIVERSE.items():
    try:
        d=cli.daily(s)
        if not len(d): print(s,'empty'); continue
        d['name']=name; d['sector']=sector
        frames.append(d)
        print(f'{s:5s} {name:10s} {len(d):5d} {d["date"].min().date()} -> {d["date"].max().date()} 换月跳空 {int(d["roll_gap"].sum())}')
    except Exception as e:
        print(s,'ERR',str(e)[:60])
df=pd.concat(frames, ignore_index=True)
df.to_parquet(CACHE, index=False)
print('total rows', len(df), 'symbols', df['symbol'].nunique())
