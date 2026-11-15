import requests, json
H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
DC='https://datacenter-web.eastmoney.com/api/data/v1/get'
for rpt in ['RPT_ECONOMY_CPI','RPT_ECONOMY_GDP','RPT_ECONOMY_PMI','RPT_ECONOMY_PPI']:
    try:
        r=requests.get(DC,params={'reportName':rpt,'columns':'ALL','pageNumber':1,'pageSize':3,'sortColumns':'REPORT_DATE','sortTypes':'-1'},headers=H,timeout=25)
        j=r.json(); d=(j.get('result') or {}).get('data') or []
        print('==',rpt, 'pages', (j.get('result') or {}).get('pages'))
        if d: print(json.dumps(d[0], ensure_ascii=False)[:600])
    except Exception as e: print(rpt,'ERR',str(e)[:80])
