"""幸存者偏差量化：交易所清单里有退市日期，但我们的面板里有没有这些股票？

步骤：
 1. 从沪深交易所清单提取「已退市」代码（DELIST_DATE 非空）；
 2. 与面板股票池比对，算出覆盖率（面板里缺多少退市股）；
 3. 用「缺的退市股在退市前的规模与行业」估计偏差方向。
"""
import sys, json, time; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import requests, pandas as pd, numpy as np
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
OUT = DATA
H={'User-Agent':'Mozilla/5.0','Referer':'https://www.sse.com.cn/'}
def sse_all():
    rows=[]; page=1
    while True:
        try:
            r=requests.get('https://query.sse.com.cn/sseQuery/commonQuery.do',
              params={'sqlId':'COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L','isPagination':'true','pageHelp.pageSize':100,
                      'pageHelp.pageNo':page,'pageHelp.beginPage':page,'pageHelp.cacheSize':1,'pageHelp.endPage':page},
              headers=H,timeout=25)
            d=r.json()['pageHelp']['data']
        except Exception as e:
            print('page',page,'err',e); time.sleep(2); continue
        if not d: break
        rows.extend(d); page+=1
        if len(d)<100: break
        time.sleep(0.05)
    return pd.DataFrame(rows)
sse=sse_all()
print('SSE rows', len(sse))
sse['delisted']=sse['DELIST_DATE'].astype(str).str.strip().replace('-',np.nan)
n_del=int(sse['delisted'].notna().sum())
print('SSE 已退市', n_del, '/', len(sse))
sse[['A_STOCK_CODE','COMPANY_ABBR','LIST_DATE','DELIST_DATE']].to_csv(OUT / 'sse_list.csv',index=False)
# 面板股票池
panel=pd.read_parquet('CACHE/a_share_daily_v2.parquet', columns=['symbol']).drop_duplicates()
pset=set(panel['symbol'].astype(str).str.zfill(6))
sse_codes=set(sse['A_STOCK_CODE'].astype(str).str.zfill(6))
del_codes=set(sse.loc[sse['delisted'].notna(),'A_STOCK_CODE'].astype(str).str.zfill(6))
print('面板覆盖的沪市代码', len(pset&sse_codes), '/', len(sse_codes))
print('面板覆盖的退市股', len(pset&del_codes), '/', len(del_codes))
miss=del_codes-pset
print('缺失退市股', len(miss))
rows=[]
for _,r in sse[sse['delisted'].notna()].iterrows():
    code=str(r['A_STOCK_CODE']).zfill(6)
    rows.append({'symbol':code,'name':r['COMPANY_ABBR'],'list_date':r['LIST_DATE'],'delist_date':r['DELIST_DATE'],
                 'in_panel':code in pset})
d=pd.DataFrame(rows)
d.to_csv(OUT / 'delisted_check.csv',index=False)
print(d['in_panel'].value_counts().to_dict())
print('按退市年份:', d.assign(y=d['delist_date'].astype(str).str[:4]).groupby('y').size().to_dict())
