"""实测传导时滞：多事件 × 多标的，看反应在第几天见顶。

结论要回答：市场是否真的「先反应上游、再传导下游」？
如果不是，暴露期就不是上游先涨，而是**同期一起动**。
"""
import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, requests, json, time
from stellarq.paths import DATA, ROOT  # noqa: E402
H={'User-Agent':'Mozilla/5.0','Referer':'https://gu.qq.com/'}
CACHE={}
def daily(code, n=1000):
    if code in CACHE: return CACHE[code]
    sym=('sh' if code.startswith(('6','68')) else 'sz')+code
    for _ in range(3):
        try:
            r=requests.get('https://ifzq.gtimg.cn/appstock/app/fqkline/get',
                           params={'param':f'{sym},day,,,{n},'}, headers=H, timeout=20)
            j=r.json(); d=j['data'][sym].get('day') or j['data'][sym].get('qfqday')
            if not d: return None
            df=pd.DataFrame(d).iloc[:,:2]; df.columns=['date','close']
            df['date']=pd.to_datetime(df['date']); df['close']=pd.to_numeric(df['close'])
            s=df.set_index('date').sort_index()['close']
            CACHE[code]=s; return s
        except Exception:
            time.sleep(1)
    return None

OUT = DATA
# 三个有明确日期的事件
EVENTS=[
 ('2026-11-09','长征十二号发射低轨29组卫星'),
 ('2026-08-19','朱雀三号陆地回收'),
 ('2026-09-01','智神星一号首飞'),
]
# 环节 → 标的
LAYERS={
 '上游器件':{'昊志机电':'300503','中天火箭':'003009','博云新材':'002297'},
 '中游总装':{'中国卫星':'600118','上海沪工':'603131'},
 '材料':{'中简科技':'300777','光威复材':'300699'},
 '算力链':{'中际旭创':'300308','新易盛':'300502','长电科技':'600584'},
}
HOR=[1,2,3,5,10,20]
BENCH=daily('399300')   # 沪深300 作基准
res={}
for edate,ename in EVENTS:
    rows={}
    for layer,stocks in LAYERS.items():
        vals=[]
        for nm,code in stocks.items():
            px=daily(code)
            if px is None: continue
            px=px.dropna()
            i=px.index.searchsorted(pd.Timestamp(edate))
            if i<=1 or i>=len(px)-max(HOR): continue
            base=px.iloc[i-1]
            row=[]
            for h in HOR:
                j=i+h-1
                if j>=len(px): row.append(np.nan); continue
                r_stock=float(px.iloc[j]/base-1)
                # 基准超额
                if BENCH is not None:
                    bi=BENCH.index.searchsorted(pd.Timestamp(edate))
                    if bi>0 and bi+h-1<len(BENCH):
                        r_bench=float(BENCH.iloc[bi+h-1]/BENCH.iloc[bi-1]-1)
                        row.append(r_stock-r_bench); continue
                row.append(r_stock)
            vals.append(row)
        if vals:
            arr=np.nanmean(np.array(vals),axis=0)
            rows[layer]={'mean':[round(float(x),4) for x in arr],'n':len(vals)}
    res[ename]={'date':edate,'layers':rows}
    print(f'\n=== {ename}（{edate}）')
    print(f'{"环节":10s} ' + ' '.join(f'{h:>7d}d' for h in HOR) + '  n  （超额 vs 沪深300）')
    for layer,v in rows.items():
        cells=' '.join(f'{x*100:>7.1f}%' for x in v['mean'])
        print(f'{layer:10s} {cells}  {v["n"]}')

# 统计：首日 vs 之后的平均
print('\n=== 全样本：事件后各期的平均超额（绝对涨跌，非超额）')
allv=np.array([v['mean'] for e in res.values() for v in e['layers'].values()])
if len(allv):
    m=np.nanmean(allv,axis=0)
    print('  ' + ' '.join(f'{h:>7d}d' for h in HOR))
    print('  ' + ' '.join(f'{x*100:>7.1f}%' for x in m))
json.dump(res, open(OUT / 'leadtime.json','w'), ensure_ascii=False, indent=1)
print('\nsaved leadtime.json')
