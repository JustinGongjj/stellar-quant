"""库存周期×行业的敏感性检验（三道闸门）。

闸门1：换相位判定窗口（3/6/9 月）
闸门2：样本切分（前后半段各自成立吗）
闸门3：不是单一年份驱动
"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.macro.wind_panel import month_end_panel
from stellarq.paths import DATA, ROOT  # noqa: E402
OUT = DATA
ind=pd.read_parquet(OUT / 'industry_index.parquet')
ind['ym']=ind['date'].dt.to_period('M')
mret=ind.groupby(['ym','industry'])['ret'].apply(lambda x:(1+x).prod()-1).unstack()
exc=mret.sub(mret.mean(axis=1), axis=0)

m=month_end_panel()
inv=(m['inventory']/m['inventory'].shift(12)-1).dropna()
prof=(m['profit_cum']/m['profit_cum'].shift(12)-1).dropna()

def sgn(s): return pd.Series(np.sign(s.to_numpy(dtype='float64')), index=s.index)
def phases(lb):
    d=pd.DataFrame({'inv':inv,'prof':prof}).dropna()
    d['di']=sgn(d['inv'].diff(lb)); d['dp']=sgn(d['prof'].diff(lb))
    def ph(r):
        a,b=r['di'],r['dp']
        if not (np.isfinite(a) and np.isfinite(b)): return None
        if a>0 and b>0: return '主动补库'
        if a>0 and b<=0: return '被动补库'
        if a<=0 and b>0: return '主动去库'
        return '被动去库'
    return d.apply(ph,axis=1)

def stat(phase_series, target_phase, label):
    common=exc.index.intersection(phase_series.dropna().index)
    mask=(phase_series.loc[common]==target_phase)
    n=int(mask.sum())
    if n<5: return None
    sub=exc.loc[common][mask]
    mu=sub.mean(); t=mu/(sub.std(ddof=1)/np.sqrt(n))
    return {'n':n,'mean':mu,'t':t,'label':label}

print('=== 闸门1：换判定窗口，结论稳吗\n')
CAND={'被动补库':['通信设备','航空装备Ⅱ'],'被动去库':['IT服务Ⅱ','计算机设备','光学光电子']}
rows=[]
for lb in (3,6,9):
    ps=phases(lb)
    for ph,names in CAND.items():
        for nm in names:
            r=stat(ps, ph, f'lb={lb}')
            if r is None: continue
            v=float(r['mean'].get(nm,np.nan))*100; tv=float(r['t'].get(nm,np.nan))
            rows.append({'lb':lb,'phase':ph,'industry':nm,'mean':round(v,2),'t':round(tv,2),'n':r['n']})
            print(f'  窗口{lb} {ph:6s} {nm:10s} 超额{v:+6.2f}% t={tv:+5.2f} (n={r["n"]})')
    print()

print('=== 闸门2：前后半段切分')
ps=phases(3)
common=exc.index.intersection(ps.dropna().index).sort_values()
half=len(common)//2
for ph,names in CAND.items():
    for nm in names:
        for seg,idx in [('前半',common[:half]),('后半',common[half:])]:
            mk=(ps.loc[idx]==ph); n=int(mk.sum())
            if n<3: print(f'  {ph} {nm} {seg}: 样本{n}不足'); continue
            v=float(exc.loc[idx][mk][nm].mean())*100
            print(f'  {ph:6s} {nm:10s} {seg} 超额{v:+6.2f}% (n={n})')
    print()

print('=== 闸门3：逐年看，是否单一年份驱动')
ps=phases(3)
for ph,names in CAND.items():
    for nm in names[:2]:
        yr={}
        for y in sorted(set(common.year)):
            idx=common[common.year==y]
            mk=(ps.loc[idx]==ph)
            if mk.sum()>=2: yr[int(y)]=round(float(exc.loc[idx][mk][nm].mean())*100,2)
        print(f'  {ph:6s} {nm:10s} 逐年: {yr}')

json.dump(rows, open(OUT / 'industry_cycle_sens.json','w'), ensure_ascii=False, indent=1)
# 判定
sig=[r for r in rows if abs(r['t'])>=2 and r['mean']>0]
same_sign=sum(1 for nm in ['通信设备','航空装备Ⅱ','IT服务Ⅱ','计算机设备','光学光电子']
              if len([r for r in rows if r['industry']==nm and r['mean']>0])==3)
print(f'\n三种窗口下都为正的行业: {same_sign}/5')
print('结论:', '相对稳健' if same_sign>=4 else '不稳定，不能作为交易依据')
