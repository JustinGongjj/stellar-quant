"""抓取行业板块指数日线，用于算行业动量与周期位置。"""
import json, time
import pandas as pd, requests
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
H={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'}

boards={}
for pn in range(1,4):
    try:
        r=requests.get('https://push2.eastmoney.com/api/qt/clist/get',
            params={'pn':pn,'pz':100,'po':1,'np':1,'fltt':2,'invt':2,'fid':'f3',
                    'fs':'m:90+t:2','fields':'f12,f14'}, headers=H, timeout=20)
        d=(r.json().get('data') or {}).get('diff') or []
    except Exception as e:
        print('ERR', str(e)[:60]); break
    if not d: break
    for x in d: boards[x['f12']]=x['f14']
print('板块:', len(boards))

rows=[]
for i,(bk,name) in enumerate(boards.items()):
    try:
        r=requests.get('https://push2his.eastmoney.com/api/qt/stock/kline/get',
            params={'secid':f'90.{bk}','klt':101,'fqt':1,'beg':'20200101','end':'20500101',
                    'fields1':'f1,f2,f3','fields2':'f51,f53'}, headers=H, timeout=20)
        d=(r.json().get('data') or {})
        kl=d.get('klines') or []
        for line in kl:
            p=line.split(',')
            if len(p)>=2: rows.append((name, p[0], float(p[1])))
    except Exception:
        pass
    if (i+1)%30==0: print(f'  {i+1}/{len(boards)}', flush=True)
    time.sleep(0.12)

df=pd.DataFrame(rows, columns=['industry','date','close'])
df['date']=pd.to_datetime(df['date'])
df.to_parquet(OUT / 'industry_index.parquet', index=False)
print(f'\n行业指数: {len(df)} 行, {df["industry"].nunique()} 个行业')
print('区间:', df['date'].min().date(), '~', df['date'].max().date())
