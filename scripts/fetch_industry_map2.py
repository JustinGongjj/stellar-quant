"""批量抓取个股→行业映射（分页拉全市场，一次拿行业字段）。"""
import json, time
import requests
OUT = DATA
H={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'}
FS=['m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048']  # 深主板/创业板/沪主板/科创板/北交所
m={}
for fs in FS:
    pn=1
    while True:
        for attempt in range(3):
            try:
                r=requests.get('https://push2.eastmoney.com/api/qt/clist/get',
                    params={'pn':pn,'pz':200,'po':1,'np':1,'fltt':2,'invt':2,'fid':'f12',
                            'fs':fs,'fields':'f12,f14,f100'}, headers=H, timeout=20)
                j=r.json(); d=(j.get('data') or {})
                diff=d.get('diff') or []
                total=d.get('total') or 0
                break
            except Exception:
                diff=[]; total=0; time.sleep(1.2)
        if not diff: break
        for x in diff:
            c=str(x.get('f12') or ''); ind=x.get('f100')
            if c and ind and ind!='-': m[c]=ind
        if pn*200>=total: break
        pn+=1
        time.sleep(0.25)
print('批量映射:', len(m))
json.dump(m, open(OUT / 'stock_industry.json','w'), ensure_ascii=False, indent=0)
from collections import Counter
from stellarq.paths import DATA  # noqa: E402
cnt=Counter(m.values())
print('行业数:', len(cnt))
print('前10:', cnt.most_common(10))
