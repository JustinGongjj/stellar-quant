"""测试缠论工作站批量调用速度，决定全市场扫描是否可行。"""
import sys, time; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from stellarq.bridge.chanlun import ChanlunStation
from concurrent.futures import ThreadPoolExecutor
from stellarq.paths import ROOT  # noqa: E402
c=ChanlunStation()
codes=['300308','300502','600183','002916','688017','300124','002050','601689','600584','002156']
t0=time.time()
ok=0
for code in codes:
    d=c.analyze(code)
    if d: ok+=1
t1=time.time()
print(f'串行 10 只: {t1-t0:.1f}s，成功 {ok}')
# 并发
t2=time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    res=list(ex.map(lambda x: c.analyze(x), codes))
t3=time.time()
print(f'并发8 10 只: {t3-t2:.1f}s，成功 {sum(1 for r in res if r)}')
per=(t3-t2)/10
print(f'\n单只平均 {per:.2f}s → 4700 只需 {per*4700/60:.1f} 分钟（并发8）')
if per*4700/60 > 60:
    print('结论：全市场扫描不可行，需改用抽样或本地复算')
