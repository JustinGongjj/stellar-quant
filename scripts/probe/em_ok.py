import requests, time
for u in ['https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.600519&fields1=f1&fields2=f51,f53&klt=101&fqt=0&lmt=3',
          'https://ifzq.gtimg.cn/appstock/app/kline/mkline?param=sh600519,m5,,3',
          'https://qt.gtimg.cn/q=sh600519']:
    try:
        r=requests.get(u, headers={'User-Agent':'Mozilla/5.0','Referer':'https://gu.qq.com/'}, timeout=15)
        print(r.status_code, u[:60], r.text[:120].replace('\n',' '))
    except Exception as e: print('ERR', u[:60], e)
