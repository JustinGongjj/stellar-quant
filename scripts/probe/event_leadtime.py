"""用事件研究法实测传导时滞：
选取已发生的产业事件（有明确日期），看上游/下游股价在事件后
第 1/3/5/10/20 日的累计超额，找出反应最快与最慢的环节。
"""
import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd, requests
H={'User-Agent':'Mozilla/5.0','Referer':'https://gu.qq.com/'}
def daily(code, n=1200):
    sym=('sh' if code.startswith(('6','68')) else 'sz')+code
    r=requests.get('https://ifzq.gtimg.cn/appstock/app/fqkline/get',
                   params={'param':f'{sym},day,,,{n},'}, headers=H, timeout=20)
    j=r.json(); d=j['data'][sym].get('day') or j['data'][sym].get('qfqday')
    if not d: return None
    df=pd.DataFrame(d).iloc[:,:2]; df.columns=['date','close']
    df['date']=pd.to_datetime(df['date']); df['close']=pd.to_numeric(df['close'])
    return df.set_index('date').sort_index()['close']

# 事件：日期 → 传闻/公告
EVENTS=[
 ('2026-09-01','智神星一号首飞',['蓝箭概念','箭体系']),
]
# 用一组有代表性的标的做事件窗反应测试
STOCKS={'中际旭创':'300308','新易盛':'300502','生益科技':'600183','深南电路':'002916',
        '绿的谐波':'688017','三花智控':'002050','拓普集团':'601689','昊志机电':'300503',
        '云南锗业':'002428','润禾材料':'300727','长电科技':'600584','寒武纪':'688256'}
def event_window(px, evdate, horizons=(1,3,5,10,20)):
    px=px.dropna()
    idx=px.index.searchsorted(pd.Timestamp(evdate))
    if idx<=0 or idx>=len(px): return None
    base=px.iloc[idx-1]
    out={}
    for h in horizons:
        j=idx+h-1
        if j<len(px): out[h]=float(px.iloc[j]/base-1)
    return out
print('事件研究：2026-09-01 星河动力智神星一号首飞\n')
rows=[]
for name,code in STOCKS.items():
    try:
        px=daily(code)
        w=event_window(px,'2026-09-01')
        if w: rows.append((name,w))
    except Exception as e:
        pass
if not rows:
    print('无数据')
else:
    print(f'{"标的":10s} ' + ' '.join(f'{h:>7d}d' for h in (1,3,5,10,20)))
    for name,w in sorted(rows, key=lambda x:-(x[1].get(20,0))):
        print(f'{name:10s} ' + ' '.join(f'{w.get(h,float("nan"))*100:>7.1f}%' for h in (1,3,5,10,20)))
