"""构建公告事件面板：逐日抓全市场公告，按关键词归类，输出 (symbol, ann_date, event, count)。

规模估算：全市场每天约 700–2000 条公告，2013–2026 约 3300 个交易日 →
约 200 万条记录、每次查询 1 页（pageSize=30）不够，需按 total 翻页（约 25–70 页/天）。
因此这里按「每周一次查询 + 翻页」的方式压缩请求数：seDate 支持区间，一次能取回整周。
"""
import sys, time, json, os; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from stellarq.data.cninfo import Cninfo, EVENT_KEYWORDS
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
OUT = DATA
CACHE='CACHE/events'
os.makedirs(CACHE, exist_ok=True)
def w(m):
    with open(OUT / 'build_events.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'build_events.log','w').close()
c=Cninfo(pause=0.15)
t0=time.time()
# 按自然月切片，避免单次 total 过大
periods=pd.period_range('2013-01','2026-09',freq='M')
for i,per in enumerate(periods):
    path=f'CACHE/{per}.parquet'
    if os.path.exists(path): continue
    start=str(per.start_time.date()); end=str(per.end_time.date())
    frames=[]
    for cat,kw in EVENT_KEYWORDS.items():
        try:
            d=c.announcements(start,end,keyword=kw,max_pages=120)
            if len(d):
                d['event']=cat
                frames.append(d[['symbol','ann_date','event','title']])
        except Exception as e:
            w(f'{per} {cat} ERR {e}')
    if frames:
        pd.concat(frames,ignore_index=True).to_parquet(path,index=False)
        w(f'{per} ok {sum(len(f) for f in frames)} {round(time.time()-t0)}s')
    else:
        pd.DataFrame(columns=['symbol','ann_date','event','title']).to_parquet(path,index=False)
        w(f'{per} empty {round(time.time()-t0)}s')
w('DONE')
