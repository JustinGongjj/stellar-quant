import requests
H={'Referer':'https://finance.sina.com.cn','User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126'}
for datalen in [1023, 3000, 8000]:
    r=requests.get('https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData',
        params={'symbol':'sz000001','scale':240,'ma':'no','datalen':datalen}, headers=H, timeout=20)
    try:
        j=r.json(); print(datalen, len(j), j[0]['day'] if j else None, j[-1]['day'] if j else None)
    except Exception:
        print(datalen, 'ERR', r.text[:100])
