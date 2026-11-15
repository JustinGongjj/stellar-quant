import requests
H={'Referer':'https://finance.sina.com.cn','User-Agent':'Mozilla/5.0'}
# 新浪期货日线
for code in ['AU0','RB0','C0','IF0','CU0']:
    try:
        u=f'https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_{code}=/InnerFuturesNewService.getDailyKLine?symbol={code}'
        r=requests.get(u, headers=H, timeout=20)
        txt=r.text
        print(code, r.status_code, len(txt), txt[:120].replace('\n',' '))
    except Exception as e:
        print(code,'ERR',str(e)[:70])
