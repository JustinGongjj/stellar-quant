import requests, time
H={'User-Agent':'Mozilla/5.0','Referer':'https://gu.qq.com/'}
tests=[
 ('腾讯 fqkline', 'https://ifzq.gtimg.cn/appstock/app/fqkline/get', {'param':'sh600519,day,2026-09-25,2026-11-14,200,'}),
 ('腾讯 web fqkline','https://web.ifzq.gtimg.cn/appstock/app/fqkline/get', {'param':'sh600519,day,2026-09-25,2026-11-14,200,'}),
 ('新浪日线','https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData', {'symbol':'sh600519','scale':240,'ma':'no','datalen':40}),
 ('东财','https://push2his.eastmoney.com/api/qt/stock/kline/get', {'secid':'1.600519','fields1':'f1','fields2':'f51,f53','klt':101,'fqt':0,'lmt':5}),
]
for name,u,p in tests:
    try:
        r=requests.get(u,params=p,headers=H,timeout=20)
        print(f'{name:14s} {r.status_code} {r.text[:120].replace(chr(10)," ")}')
    except Exception as e:
        print(f'{name:14s} ERR {str(e)[:70]}')
