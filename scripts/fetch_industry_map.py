"""抓取个股→行业映射 + 行业指数历史（东财板块）。

产出：
  reports/data/stock_industry.json   个股代码 → 行业名
  reports/data/industry_index.parquet 行业指数日线（用于算行业动量与周期位置）
"""
import json, time, sys
__import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
OUT = DATA
H={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'}
C='CACHE/'; R=C+'wide2/'
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
valid=np.load(R+'wide_valid.npy'); last=len(valid)-1
while last>0 and valid[last].sum()<500: last-=1
codes=sorted(set(syms[np.asarray(valid[last],dtype=bool)].tolist()))
print('股票数:', len(codes))

def secid(code):
    return ('1.' if code.startswith(('6','9')) else '0.')+code

def get_ind(code):
    for _ in range(2):
        try:
            r=requests.get('https://push2.eastmoney.com/api/qt/stock/get',
                params={'secid':secid(code),'fields':'f127'}, headers=H, timeout=12)
            d=r.json().get('data') or {}
            return code, d.get('f127')
        except Exception:
            time.sleep(0.4)
    return code, None

m={}
with ThreadPoolExecutor(max_workers=16) as ex:
    for i,(code,ind) in enumerate(ex.map(get_ind, codes)):
        if ind: m[code]=ind
        if (i+1)%1500==0: print(f'  {i+1}/{len(codes)} 已映射 {len(m)}', flush=True)
json.dump(m, open(OUT / 'stock_industry.json','w'), ensure_ascii=False, indent=0)
from collections import Counter
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
cnt=Counter(m.values())
print(f'\n映射 {len(m)} 只，行业 {len(cnt)} 个')
print('最大行业:', cnt.most_common(8))
