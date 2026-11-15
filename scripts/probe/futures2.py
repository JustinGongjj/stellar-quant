import requests
H={'Referer':'https://gu.qq.com/','User-Agent':'Mozilla/5.0'}
for p in ['IF0,day,,,10,','CFF_RE_IF,day,,,10,','nf_IF0,day,,,10,','CFF_RE_IF2601,day,,,10,','sh000300,day,,,2,']:
    try:
        r=requests.get('https://web.ifzq.gtimg.cn/appstock/app/fqkline/get',params={'param':p},headers=H,timeout=20)
        t=r.text[:160].replace('\n',' ')
        print(p, r.status_code, t)
    except Exception as e: print(p,'ERR',str(e)[:60])
