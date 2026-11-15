"""探测行业分类数据源。"""
import requests, json
H={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'}
# 1) 行业板块列表
try:
    r=requests.get('https://push2.eastmoney.com/api/qt/clist/get',
        params={'pn':1,'pz':200,'po':1,'np':1,'fltt':2,'invt':2,'fid':'f3',
                'fs':'m:90+t:2','fields':'f12,f14,f3,f62'}, headers=H, timeout=20)
    j=r.json()
    d=(j.get('data') or {}).get('diff') or []
    print(f'行业板块: {len(d)} 个')
    for x in d[:5]: print('  ', x.get('f12'), x.get('f14'), x.get('f3'))
except Exception as e:
    print('板块列表 ERR', str(e)[:80])
# 2) 个股所属行业
try:
    r=requests.get('https://push2.eastmoney.com/api/qt/stock/get',
        params={'secid':'0.300308','fields':'f57,f58,f127,f128'}, headers=H, timeout=15)
    print('个股行业:', r.json().get('data'))
except Exception as e:
    print('个股 ERR', str(e)[:80])
