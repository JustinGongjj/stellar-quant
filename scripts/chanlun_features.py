"""全市场缠论技术面特征提取（调缠论工作站 /api/chan）。

产出 reports/data/chanlun_feats.parquet：每只股票一行缠论结构特征。
"""
import sys, json, time; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from stellarq.bridge.chanlun import ChanlunStation
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

OUT = DATA
C='CACHE/'; R=C+'wide2/'
syms=[str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)]
valid=np.load(R+'wide_valid.npy')
last=len(valid)-1
while last>0 and valid[last].sum()<500: last-=1
codes=sorted(set(np.asarray(syms)[np.asarray(valid[last],dtype=bool)].tolist()))
print(f'目标 {len(codes)} 只')

c=ChanlunStation(timeout=20)
rows=[]; fail=0
t0=time.time()
def one(code):
    d=c.analyze(code, limit=300)
    if not d: return None
    f=c.features(d)
    f['code']=code
    f['name']=d.get('name')
    return f

with ThreadPoolExecutor(max_workers=12) as ex:
    futs={ex.submit(one,code):code for code in codes}
    done=0
    for f in as_completed(futs):
        done+=1
        r=f.result()
        if r is None: fail+=1
        else: rows.append(r)
        if done%800==0:
            print(f'  {done}/{len(codes)} ok={len(rows)} fail={fail} {time.time()-t0:.0f}s', flush=True)

df=pd.DataFrame(rows)
df.to_parquet(OUT / 'chanlun_feats.parquet', index=False)
print(f'\n完成 {len(df)} 只，失败 {fail}，用时 {time.time()-t0:.0f}s')
print('特征列:', [c for c in df.columns if c.startswith('ch_')])
print('\n走势类型分布:')
print(df['ch_trend'].value_counts().to_dict())
print('中枢位置分布:')
print(df['ch_pos'].value_counts().to_dict())
sig=df['ch_last_signal'].value_counts().to_dict()
print('买卖点分布:', sig)
json.dump({'n':len(df),'fail':fail,'trend':df['ch_trend'].value_counts().to_dict(),
           'pos':df['ch_pos'].value_counts().to_dict(),'signal':sig},
          open(OUT / 'chanlun_feats_info.json','w'), ensure_ascii=False, indent=1)
