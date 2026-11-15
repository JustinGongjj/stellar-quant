"""数据一致性核查：EM 与腾讯在不复权收盘价上是否一致（用样本股票）。"""
import sys, time; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import pandas as pd, requests
from stellarq.data.tencent import Tencent
t=Tencent()
def em(sym):
    ex='1' if sym.startswith('6') else '0'
    for _ in range(3):
        try:
            r=requests.get('https://push2his.eastmoney.com/api/qt/stock/kline/get', params={'secid':f'{ex}.{sym}','fields1':'f1','fields2':'f51,f53,f59','klt':101,'fqt':0,'beg':'20130101','end':'20500101'}, headers={'User-Agent':'Mozilla/5.0'}, timeout=25)
            j=r.json()
            if j.get('data'): return {x.split(',')[0]:float(x.split(',')[1]) for x in j['data']['klines']}
        except Exception: pass
        time.sleep(2)
    return {}
for sym in ['600519','000001','300750']:
    e=em(sym)
    try: d=t.daily(sym)
    except Exception as exc: print(sym,'tencent blocked',exc); continue
    m=d[['date','close']].copy(); m['date']=m['date'].dt.strftime('%Y-%m-%d')
    m['em']=m['date'].map(e); m=m.dropna(subset=['em'])
    if m.empty: print(sym,'no overlap'); continue
    diff=(m['close']-m['em']).abs()
    print(f'{sym}: 重叠 {len(m)} 天 | 收盘差异 >0.011 的天数 {int((diff>0.011).sum())} | 最大差异 {diff.max():.2f}')
