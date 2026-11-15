"""探测社融/M2 数据源。"""
import requests, json
H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
# 东财宏观接口：社融(RPT_ECONOMY_SOCIAL_FINANCING)、M2(RPT_ECONOMY_CURRENCY_SUPPLY)
tests={
 '社融': ('RPT_ECONOMY_SOCIAL_FINANCING', {'columns':'ALL'}),
 'M2':   ('RPT_ECONOMY_CURRENCY_SUPPLY', {'columns':'ALL'}),
 '新增贷款': ('RPT_ECONOMY_LOAN', {'columns':'ALL'}),
}
for name,(rpt,params) in tests.items():
    p={'reportName':rpt,'columns':'ALL','pageNumber':1,'pageSize':5,'sortColumns':'REPORT_DATE','sortTypes':'-1'}
    try:
        r=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=p,headers=H,timeout=20)
        j=r.json()
        ok=j.get('success'); data=(j.get('result') or {}).get('data') or []
        print(f'{name:8s} ok={ok} rows={len(data)}')
        if data: print('   字段:', list(data[0].keys())); print('   样例:', {k:data[0][k] for k in list(data[0])[:5]})
    except Exception as e:
        print(f'{name:8s} ERR {str(e)[:60]}')
