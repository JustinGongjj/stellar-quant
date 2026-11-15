import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import requests, json
H={'Referer':'https://gu.qq.com/','User-Agent':'Mozilla/5.0'}
tests=[
 ('https://web.ifzq.gtimg.cn/appstock/app/fqkline/get', {'param':'CFF_RE_IF0,day,,,10,'}),
 ('https://ifzq.gtimg.cn/appstock/app/fqkline/get', {'param':'CFF_RE_IF0,day,,,10,'}),
 ('https://web.ifzq.gtimg.cn/appstock/app/fqkline/get', {'param':'sh000905,day,,,3,'}),
]
for u,p in tests:
    try:
        r=requests.get(u,params=p,headers=H,timeout=20)
        print(p['param'], r.status_code, r.text[:220].replace('\n',' '))
    except Exception as e: print(p['param'],'ERR',str(e)[:80])
