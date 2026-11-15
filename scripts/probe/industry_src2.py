import requests, json, pandas as pd
H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
# 1) 东财财务接口里的行业字段 BOARD_NAME
r=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',
  params={'reportName':'RPT_LICO_FN_CPD','columns':'SECURITY_CODE,BOARD_NAME,PUBLISHNAME,REPORTDATE','pageNumber':1,'pageSize':3},headers=H,timeout=25)
print('BOARD_NAME', str(r.json())[:300])
# 2) 证监会行业分类（巨潮）
r2=requests.get('http://www.cninfo.com.cn/new/data/szse_stock.json', headers={'User-Agent':'Mozilla/5.0','Referer':'http://www.cninfo.com.cn/'}, timeout=25)
print('szse_stock', r2.status_code, r2.text[:200])
