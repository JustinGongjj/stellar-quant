"""用本地面板自建行业指数（等权），比外部接口更可控。

输入：stock_industry.json（1482 只覆盖）+ 面板日线
输出：industry_index.parquet + stock_industry_full.json
"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from collections import Counter
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
OUT = DATA
C='CACHE/'; R=C+'wide2/'
cl=np.load(R+'wide_close.npy',mmap_mode='r')
dates=pd.to_datetime(np.load(R+'wide_dates.npy',allow_pickle=True))
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
valid=np.load(R+'wide_valid.npy')

ind=json.load(open(OUT / 'stock_industry.json',encoding='utf-8'))
# 只保留有 >=8 只成分的行业（太少没有统计意义）
cnt=Counter(ind.values())
keep={k for k,v in cnt.items() if v>=8}
print(f'行业 {len(cnt)} 个，保留 >=8 只的 {len(keep)} 个')

# 逐行业算等权日收益
groups={}
for c,i in ind.items():
    if i in keep: groups.setdefault(i,[]).append(c)
sym2col={s:j for j,s in enumerate(syms)}

# 取最近 1200 个交易日
t0=max(0,len(dates)-1200)
recs=[]
for name, members in groups.items():
    cols=[sym2col[s] for s in members if s in sym2col]
    if len(cols)<8: continue
    sub=np.asarray(cl[t0:, cols],dtype='float32')     # (T, n)
    v=np.asarray(valid[t0:, cols])
    prev=np.vstack([sub[:1], sub[:-1]])
    ret=np.where(v & (prev>0), sub/prev-1, np.nan)
    eq=np.nanmean(ret, axis=1)
    for k in range(len(eq)):
        recs.append((name, dates[t0+k], float(eq[k])))

df=pd.DataFrame(recs, columns=['industry','date','ret'])
df=df.dropna()
df.to_parquet(OUT / 'industry_index.parquet', index=False)
print(f'行业指数: {len(df)} 行, {df["industry"].nunique()} 行业')
print('区间:', df['date'].min().date(), '~', df['date'].max().date())
json.dump({k:sorted(v) for k,v in groups.items() if len(v)>=8},
          open(OUT / 'industry_members.json','w'), ensure_ascii=False, indent=0)
print('成分表已存')
