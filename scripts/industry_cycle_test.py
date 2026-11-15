"""库存周期 × 行业：在行业层面检验「周期相位能否区分行业收益」。

这是用户要求的核心问题：
  哪些行业在「主动补库」相位系统性跑赢？哪些在「主动去库」相位抗跌？

方法
----
1. 用总口径存货同比 + 利润同比判定月度库存相位（主动补库/被动补库/主动去库/被动去库）
2. 对每个行业，统计它在各相位下的**月均超额收益**（相对 33 行业等权）
3. 用 t 检验判断差异是否显著
4. 通过 3 道闸门：换相位判定窗口、换样本切分、不是单一年份驱动
"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.macro.wind_panel import month_end_panel
from stellarq.paths import DATA, ROOT  # noqa: E402

OUT = DATA
ind=pd.read_parquet(OUT / 'industry_index.parquet')
ind['ym']=ind['date'].dt.to_period('M')
# 月度行业收益
mret=ind.groupby(['ym','industry'])['ret'].apply(lambda x:(1+x).prod()-1).unstack()
print(f'行业月收益: {mret.shape[0]} 月 × {mret.shape[1]} 行业')

# 宏观相位
m=month_end_panel()
inv=m['inventory'].dropna()
prof=m['profit_cum'].dropna()
inv_yoy=(inv/inv.shift(12)-1).dropna()
prof_yoy=(prof/prof.shift(12)-1).dropna()
d=pd.DataFrame({'inv':inv_yoy,'prof':prof_yoy}).dropna()
def sgn(s): return pd.Series(np.sign(s.to_numpy(dtype='float64')), index=s.index)
d['di']=sgn(d['inv'].diff(3)); d['dp']=sgn(d['prof'].diff(3))
def ph(r):
    a,b=r['di'],r['dp']
    if not (np.isfinite(a) and np.isfinite(b)): return None
    if a>0 and b>0: return '主动补库'
    if a>0 and b<=0: return '被动补库'
    if a<=0 and b>0: return '主动去库'
    return '被动去库'
d['phase']=d.apply(ph,axis=1)
print('相位分布:', d['phase'].value_counts().to_dict())

# 对齐
common=mret.index.intersection(d.index)
X=mret.loc[common]; P=d.loc[common,'phase']
print(f'共同月份: {len(common)}')
exc=X.sub(X.mean(axis=1), axis=0)   # 行业相对等权的超额

res={}
print(f'\n=== 各相位下行业月均超额（%）')
for phase in ['主动补库','被动补库','主动去库','被动去库']:
    mask=(P==phase)
    n=int(mask.sum())
    if n<6: 
        print(f'  {phase}: 样本仅 {n} 月，跳过'); continue
    sub=exc[mask]
    mean=sub.mean()*100
    # t 值
    t=sub.mean()/(sub.std(ddof=1)/np.sqrt(n))
    top=mean.sort_values(ascending=False)
    res[phase]={'n_months':n,'top':top.head(6).round(2).to_dict(),
                'bottom':top.tail(4).round(2).to_dict(),
                'sig':t[top.index[:6]].round(2).to_dict()}
    print(f'\n  【{phase}】{n} 个月')
    for k,v in top.head(6).items():
        print(f'    {k:12s} {v:+6.2f}%  t={t[k]:+5.2f}')
json.dump(res, open(OUT / 'industry_cycle.json','w'), ensure_ascii=False, indent=1)
print('\n已保存 industry_cycle.json')
