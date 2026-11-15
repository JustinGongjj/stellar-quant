import requests
H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
cands=['RPT_ECONOMY_SOCIAL_FINANCING','RPT_ECONOMY_SOCIETY_FINANCING','RPT_ECONOMY_TOTAL_RETAIL',
       'RPT_ECONOMY_SHRZ','RPT_ECONOMY_SOCIAL_FINANCE','RPT_ECONOMY_MONEY_SUPPLY']
for rpt in cands:
    p={'reportName':rpt,'columns':'ALL','pageNumber':1,'pageSize':3,'sortColumns':'REPORT_DATE','sortTypes':'-1'}
    try:
        r=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=p,headers=H,timeout=15)
        j=r.json()
        if j.get('success'):
            d=(j.get('result') or {}).get('data') or []
            print(f'{rpt:32s} OK rows={len(d)} 字段={list(d[0].keys())[:8] if d else []}')
        else:
            print(f'{rpt:32s} no')
    except Exception as e:
        print(f'{rpt:32s} ERR')
