"""检查能否用行情数据实测「传导时滞」。"""
import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np, pandas as pd, requests
H={'User-Agent':'Mozilla/5.0','Referer':'https://gu.qq.com/'}
def daily(code, n=900):
    sym=('sh' if code.startswith(('6','68')) else 'sz')+code
    r=requests.get('https://ifzq.gtimg.cn/appstock/app/fqkline/get',
                   params={'param':f'{sym},day,,,{n},'}, headers=H, timeout=20)
    j=r.json(); d=j['data'][sym].get('day') or j['data'][sym].get('qfqday')
    if not d: return None
    df=pd.DataFrame(d)
    df=df.iloc[:, :2]; df.columns=['date','close']
    df['date']=pd.to_datetime(df['date']); df['close']=pd.to_numeric(df['close'])
    return df.set_index('date').sort_index()
pairs=[('300308','300502','中际旭创','新易盛'),
       ('600183','002916','生益科技','深南电路'),
       ('300777','600862','中简科技','中航高科')]
for a,b,na,nb in pairs:
    try:
        da=daily(a); db=daily(b)
        if da is None or db is None: print(f'{na}/{nb}: 缺失'); continue
        j=pd.concat([da['close'].pct_change(), db['close'].pct_change()],
                    axis=1, keys=['a','b']).dropna()
        c=j['a'].corr(j['b'])
        best=(0,c)
        for k in range(1,11):
            cl=j['a'].corr(j['b'].shift(-k))   # a 领先 b k 天
            if cl>best[1]: best=(k,cl)
        print(f'{na}/{nb}: {len(j)}天 同期{c:.3f} | 最优领先 {best[0]}天 相关 {best[1]:.3f}')
    except Exception as e:
        print(f'{na}/{nb}: ERR {str(e)[:70]}')
