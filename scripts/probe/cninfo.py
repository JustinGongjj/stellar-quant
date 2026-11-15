"""巨潮资讯公告接口探测（事件因子数据源）。"""
import requests, json
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126','Referer':'http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice','X-Requested-With':'XMLHttpRequest','Content-Type':'application/x-www-form-urlencoded; charset=UTF-8'}
s=requests.Session()
# 先访问首页拿 cookie
try:
    s.get('http://www.cninfo.com.cn/new/index', headers={'User-Agent':H['User-Agent']}, timeout=20)
    print('home ok', len(s.cookies))
except Exception as e:
    print('home ERR', e)
for params in [
    {'pageNum':1,'pageSize':5,'column':'szse','tabName':'fulltext','stock':'','seDate':'2024-04-01~2024-04-03','sortName':'','sortType':'','isHLtitle':'true'},
    {'pageNum':1,'pageSize':5,'column':'szse','tabName':'fulltext','searchkey':'业绩预告','seDate':'2024-04-01~2024-04-03'},
]:
    try:
        r=s.post('http://www.cninfo.com.cn/new/hisAnnouncement/query', data=params, headers=H, timeout=25)
        j=r.json()
        print(r.status_code, 'total', j.get('totalAnnouncement'), 'n', len(j.get('announcements') or []))
        for a in (j.get('announcements') or [])[:3]:
            print('   ', a.get('secCode'), a.get('secName'), a.get('announcementTitle','')[:40], a.get('adjunctUrl'))
    except Exception as e:
        print('query ERR', type(e).__name__, str(e)[:120], r.text[:150] if 'r' in dir() else '')
