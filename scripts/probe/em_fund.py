import requests, json
H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
url='https://datacenter-web.eastmoney.com/api/data/v1/get'
p={'reportName':'RPT_LICO_FN_CPD','columns':'ALL','pageNumber':1,'pageSize':2,'sortColumns':'UPDATE_DATE','sortTypes':'-1'}
r=requests.get(url,params=p,headers=H,timeout=25); j=r.json()
print(j.get('success'), j.get('message'))
d=(j.get('result') or {}).get('data') or []
if d:
    for k,v in list(d[0].items()): print(' ', k, '=', v)
