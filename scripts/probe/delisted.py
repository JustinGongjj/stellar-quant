import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import requests, pandas as pd
H={'User-Agent':'Mozilla/5.0','Referer':'https://www.sse.com.cn/'}
r=requests.get('https://query.sse.com.cn/sseQuery/commonQuery.do',
  params={'sqlId':'COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L','isPagination':'true','pageHelp.pageSize':100,
          'pageHelp.pageNo':1,'pageHelp.beginPage':1,'pageHelp.cacheSize':1,'pageHelp.endPage':1},headers=H,timeout=25)
d=r.json()['pageHelp']['data']
print('fields', list(d[0].keys()))
df=pd.DataFrame(d)
print(df[['A_STOCK_CODE','COMPANY_ABBR','LIST_DATE','DELIST_DATE','LIST_STATE_DESC']].head(8).to_string() if 'DELIST_DATE' in df.columns else df.head(2).to_string())
