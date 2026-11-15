"""替代方案：用「板块级事件」构造足够样本。

思路：产业事件（发射、首飞）一年只有几次，样本太少。
改用**高频可观测的信号事件**来检验"T+1 卖出"规律：
  * 概念板块指数单日大涨 >5%（相当于"事件驱动拉升"）
  * 看之后 5/10/20 日的超额收益
  
这样能拿到几百个样本，统计上才有意义。
"""
import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import pandas as pd, numpy as np, requests
H={'User-Agent':'Mozilla/5.0','Referer':'https://gu.qq.com/'}
def daily(code, n=1500):
    sym=('sh' if code.startswith(('6','68')) else 'sz')+code
    r=requests.get('https://ifzq.gtimg.cn/appstock/app/fqkline/get',
                   params={'param':f'{sym},day,,,{n},'}, headers=H, timeout=20)
    j=r.json(); d=j['data'][sym].get('day') or j['data'][sym].get('qfqday')
    if not d: return None
    df=pd.DataFrame(d).iloc[:,:2]; df.columns=['date','close']
    df['date']=pd.to_datetime(df['date']); df['close']=pd.to_numeric(df['close'])
    return df.set_index('date').sort_index()['close']
bench=daily('399300')
print('基准', None if bench is None else len(bench))
for code,nm in [('300308','中际旭创'),('300502','新易盛'),('688017','绿的谐波'),('300503','昊志机电'),('002428','云南锗业')]:
    s=daily(code)
    print(f'{nm:8s}', None if s is None else f'{len(s)} 天 {s.index.min().date()}~{s.index.max().date()}')
