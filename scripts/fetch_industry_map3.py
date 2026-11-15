"""合并多路行业映射，补全覆盖率。

路1：东财 clist 分页（f100 行业）
路2：东财 stock/get 单只（f127 行业）—— 慢但准
对未覆盖的股票做单只补齐，加重试与退避。
"""
import json, time, sys
__import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor
OUT = DATA
H={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'}
C='CACHE/'; R=C+'wide2/'
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
valid=np.load(R+'wide_valid.npy'); last=len(valid)-1
while last>0 and valid[last].sum()<500: last-=1
codes=sorted(set(syms[np.asarray(valid[last],dtype=bool)].tolist()))

m=json.load(open(OUT / 'stock_industry.json',encoding='utf-8'))
print('已有:', len(m))
missing=[c for c in codes if c not in m]
print('待补:', len(missing))

def secid(code): return ('1.' if code.startswith(('6','9')) else '0.')+code
def get_ind(code):
    for k in range(3):
        try:
            r=requests.get('https://push2.eastmoney.com/api/qt/stock/get',
                params={'secid':secid(code),'fields':'f127'}, headers=H, timeout=12)
            d=r.json().get('data') or {}
            v=d.get('f127')
            if v and v!='-': return code, v
        except Exception:
            pass
        time.sleep(0.5*(k+1))
    return code, None

ok=0
with ThreadPoolExecutor(max_workers=8) as ex:
    for i,(c,v) in enumerate(ex.map(get_ind, missing)):
        if v: m[c]=v; ok+=1
        if (i+1)%800==0: print(f'  {i+1}/{len(missing)} 补到 {ok}', flush=True)
json.dump(m, open(OUT / 'stock_industry.json','w'), ensure_ascii=False, indent=0)
from collections import Counter
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
cnt=Counter(m.values())
print(f'\n最终映射 {len(m)}/{len(codes)} = {len(m)/len(codes)*100:.1f}%，行业 {len(cnt)} 个')
