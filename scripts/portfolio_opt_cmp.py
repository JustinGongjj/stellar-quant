"""三档组合在同一组日期上对比（消除样本错配）+ 优化器缺陷诊断。"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.paths import DATA, ROOT  # noqa: E402
OUT = DATA
a=pd.read_csv(OUT / 'portfolio_opt_A.csv'); b=pd.read_csv(OUT / 'portfolio_opt_B.csv'); c=pd.read_csv(OUT / 'portfolio_opt_C.csv')
common=sorted(set(a['date'])&set(b['date'])&set(c['date']))
def stats(df, name):
    d=df[df['date'].isin(common)].sort_values('date')
    ov=d['net'].rolling(5).mean().dropna(); ex=(d['net']-d['bench']).rolling(5).mean().dropna()
    nav=(1+ov).cumprod()
    return {'name':name,'n':len(d),'ann_net':float(ov.mean()*10.08),'ann_excess':float(ex.mean()*10.08),
            'sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
            'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),
            'turnover':float(d['turn'].mean()),'max_dd':float((nav/nav.cummax()-1).min())}
res=[stats(a,'A_等权top50'),stats(b,'B_个股上限3%'),stats(c,'C_行业20%+中性')]
df=pd.DataFrame(res)
print(df.round(4).to_string(index=False))
df.to_csv(OUT / 'portfolio_opt_cmp.csv',index=False)
# 诊断：B 的有效持仓数
print('\nB 持仓数分布:', b['n'].describe().to_dict())
print('A 持仓数分布:', a['n'].describe().to_dict())
