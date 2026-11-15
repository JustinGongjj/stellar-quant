import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import requests, pandas as pd, glob
H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
# 东财行业板块成分（申万/东财一级）
url='https://push2.eastmoney.com/api/qt/clist/get'
p={'pn':1,'pz':100,'po':1,'np':1,'fltt':2,'invt':2,'fid':'f3','fs':'m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23','fields':'f12,f14'}
for src in [('https://push2.eastmoney.com/api/qt/clist/get',p)]:
    try:
        r=requests.get(src[0],params=src[1],headers=H,timeout=20); j=r.json()
        print('clist', str(j)[:150])
    except Exception as e: print('ERR', e)
# 用已落盘的财务数据里的行业字段（BOARD_NAME 之前抓过吗）
fs=sorted(glob.glob(str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/fund/*.parquet'))
d=pd.read_parquet(fs[-5])
print('cols', [c for c in d.columns if 'BOARD' in c or 'INDUSTRY' in c or 'TRADE' in c])
print(d[['SECURITY_CODE','SECURITY_NAME_ABBR']].head(3).to_dict('records'))
