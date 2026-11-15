import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
from stellarq.data.cninfo import Cninfo
c=Cninfo()
for col in ('szse','sse'):
    c.session.post('http://www.cninfo.com.cn/new/hisAnnouncement/query', data={}, timeout=5) if False else None
    d=c._query(1,30,'2024-04-01~2024-04-01',column=col)
    print(col,'total',d.get('totalAnnouncement'))
# 全市场（含沪市）用 column='szse' 时是否包含 600 开头
d=c.announcements('2024-04-01','2024-04-01')
print('one day all', len(d), 'has 6xxxxx', int(d['symbol'].str.startswith('6').sum()), 'has 3/0/8', int(d['symbol'].str.startswith(('3','0','8')).sum()))
