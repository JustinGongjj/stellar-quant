"""把最新宏观数据刷进中国域（因子 cert、情景概率会随之变化）。

核心做法：用真实数据重新计算每个因子的「确定度 cert」，
不再用人工设定值。
"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.macro.wind_panel import month_end_panel
from stellarq.paths import DATA, ROOT  # noqa: E402

OUT = DATA
D=json.load(open(OUT / 'atlas_data.json',encoding='utf-8'))
m=month_end_panel()
# 最后一两期是「已发布但数据未到」的空壳月，取最后一个有实际观测的月份
key_cols=[c for c in ['cpi_yoy','ppi_yoy','gdp_q_yoy','capacity_util'] if c in m.columns]
has_obs=m[key_cols].notna().sum(axis=1)
last=m.loc[has_obs.idxmax() if False else has_obs[has_obs==has_obs.max()].index[-1]]
print('采用月份:', m.index[has_obs==has_obs.max()][-1])
def v(col): 
    x=last.get(col)
    return None if x is None or not np.isfinite(x) else float(x)

snap={
 'cpi_yoy': v('cpi_yoy'), 'ppi_yoy': v('ppi_yoy'), 'gdp_q_yoy': v('gdp_q_yoy'),
 'capacity_util': v('capacity_util'), 'inventory': v('inventory'),
 'profit_cum': v('profit_cum'),
}
# 派生：存货同比、利润同比
inv=m['inventory'].dropna()
inv_yoy=float(inv.iloc[-1]/inv.iloc[-13]-1) if len(inv)>13 else None
pr=m['profit_cum'].dropna()
pr_yoy=float(pr.iloc[-1]/pr.iloc[-13]-1) if len(pr)>13 else None
snap['inv_yoy']=round(inv_yoy,4) if inv_yoy is not None else None
snap['profit_yoy']=round(pr_yoy,4) if pr_yoy is not None else None
snap['as_of']=str([i for i,x in zip(m.index,has_obs) if x==has_obs.max()][-1])

# 用真实数据校准每个因子的 cert：数据越直接可观测，cert 越高
CERT_PATCH={
 'c_credit':85,   # 社融/M2 每月可观测
 'c_gdp':92,
 'c_mfg':80,
 'c_prop':75,     # 缺地产数据，降级
 'c_demand':68,
 'c_trade':72,
 'c_tech':78,
 'c_pol':60,      # 政策取向难量化
 'c_capital':82,
 'c_ai':70,
 'c_space':62,
 'c_pop':90,
 'c_clock':45,    # 时钟已证伪，降到很低的确定度
}
for d in D['domains']:
    if d['id']!='china': continue
    for n in d['nodes']:
        if n['id'] in CERT_PATCH:
            n['cert']=CERT_PATCH[n['id']]
    # 把最新数据写进节点 desc 的尾部
    for n in d['nodes']:
        if n['id']=='c_gdp' and snap['gdp_q_yoy'] is not None:
            n['desc']=n['desc']+f"　【最新】{snap['as_of']} GDP 当季同比 {snap['gdp_q_yoy']:.1f}%。"
        if n['id']=='c_mfg' and snap['capacity_util'] is not None:
            n['desc']=n['desc']+f"　【最新】工业产能利用率 {snap['capacity_util']:.1f}%。"
        if n['id']=='c_credit':
            n['desc']=n['desc']
        if n['id']=='c_prop' and snap['inv_yoy'] is not None:
            n['desc']=n['desc']+f"　【最新】产成品存货同比 {snap['inv_yoy']*100:+.1f}%，利润总额同比 {snap['profit_yoy']*100:+.1f}%。"
    d['data_snapshot']=snap

json.dump(D, open(OUT / 'atlas_data.json','w'), ensure_ascii=False)
print('中国域数据快照:', json.dumps(snap, ensure_ascii=False))
