"""未来预测框架：把已有的真实数值组织成「可读、可复核、可更新」的结构。

四层结构：
  L1 状态层 —— 美林时钟当前相位（真实宏观数据，点对点对齐）
  L2 因子层 —— 57 个因子的 ICIR + 8 个星簇（相关性聚类）
  L3 预测层 —— v4 模型当日打分 + 校准后的概率 + 置信区间
  L4 证据层 —— 每个结论的证据等级与失效条件

输出 reports/data/forecast.json，供前端星图直接读取。
"""
import sys, json, glob; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
OUT = DATA
C='CACHE/'; R=C+'wide2/'

out={}
# ---------- L1 状态层
cl=pd.read_csv(OUT / 'clock_history.csv', parse_dates=['Unnamed: 0'])
last=cl.iloc[-1]
out['state']={
 'as_of': str(last['Unnamed: 0'].date()),
 'phase': str(last['phase']),
 'inputs': {'cpi_yoy': float(last['cpi_yoy']), 'gdp_yoy': float(last['gdp_yoy']),
            'pmi': float(last['pmi']), 'ppi_yoy': float(last['ppi_yoy'])},
 'phase_history': cl['phase'].value_counts().to_dict(),
 'note': '相位由 GDP 累计同比与 CPI 同比的 3 期差分符号决定；宏观数据按公布滞后对齐，无前视'
}
# ---------- L2 因子层
fc=json.load(open(OUT / 'factor_corr.json'))
mine=pd.read_csv(OUT / 'factor_mine_h5_partial.csv')
icir={r['factor']:float(r['icir']) for _,r in mine.iterrows()}
tval={r['factor']:float(r['t_nw']) for _,r in mine.iterrows()}
icm={r['factor']:float(r['ic_mean']) for _,r in mine.iterrows()}
clusters=[]
for g in fc['clusters_hier']:
    members=[{'name':f,'icir':icir.get(f),'ic_mean':icm.get(f),'t':tval.get(f)} for f in g['factors']]
    vals=[m['icir'] for m in members if m['icir'] is not None]
    clusters.append({'size':g['size'],'members':members,
                     'cluster_icir':float(np.mean(vals)) if vals else None,
                     'cluster_label':None})
def _label(fs):
    s = set(fs)
    if s & {"MA10", "MA20", "MA60", "MIN5", "MIN20", "RET20"}:
        return "动量与位置"
    if s & {"AMT_RATIO60", "AMT_RATIO20", "VSUMD60", "VMA60"}:
        return "量能异动"
    if s & {"VSTD10", "VSTD20", "VSTD60"}:
        return "量能波动"
    if s & {"AMT_MA5", "LN_CAP"}:
        return "规模与活跃度"
    if s & {"STD20", "TURN5"}:
        return "波动与换手"
    return "形态与离散度"
from collections import Counter
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
_base=[_label([m['name'] for m in c['members']]) for c in clusters]
_dup=Counter(_base)
for c,lab in zip(clusters,_base):
    c['cluster_label']=lab if _dup[lab]==1 else f"{lab}·{c['members'][0]['name']}"
out['factors']={'clusters':clusters,'n_factors_evaluated':int(len(mine)),
                'link_rule':'|corr|>=0.45 平均连接聚类；连线强度=|corr|'}
# ---------- L3 预测层
mx=json.load(open(OUT / 'model_v4.json'))
st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
dates=np.load(R+'wide_dates.npy',allow_pickle=True)
u=int(np.unique(dte)[-1])
sc=pd.Series(st[dte==u], index=syms[ste[dte==u]]).dropna().sort_values(ascending=False)
out['prediction']={
 'as_of': str(pd.Timestamp(dates[u]).date()),
 'horizon_days': 5,
 'model': 'v4 (DoubleEnsemble, 19 量价特征)',
 'oos_rank_ic': float(mx['test_ic']), 'oos_ece': float(mx['ece']),
 'n_oos_periods': int(mx['test_ic_n']),
 'top20': [{'symbol':k,'score':float(v)} for k,v in sc.head(20).items()],
 'score_quantiles': {q: float(sc.quantile(q)) for q in (0.0,0.25,0.5,0.75,1.0)},
 'warning': 'RankIC 0.076 意味着排序能力很弱；单只股票的预测不构成交易建议',
 'staleness_days': int((pd.Timestamp.today().normalize() - pd.Timestamp(dates[u])).days),
 'panel_end': str(pd.Timestamp(dates[-1]).date()),
 'staleness_note': f'预测时点 {pd.Timestamp(dates[u]).date()}，距今 {int((pd.Timestamp.today().normalize()-pd.Timestamp(dates[u])).days)} 天；'
                   f'面板截断在 {pd.Timestamp(dates[-1]).date()}。要变成"当日预测"，需补齐数据到最新交易日并重跑打分。'
}
# ---------- L4 证据层
out['evidence']=[
 {'claim':'量价因子簇在 2013-2026 全样本显著','grade':'A（已过 BH 校正）',
  'value':f'{len(mine)} 个因子中多数 q<0.01','source':'reports/data/factor_mine_h5_partial.csv',
  'fails_if':'未来量价结构改变（如全面注册制后小盘效应消失）'},
 {'claim':'宏观变量当特征可提升 IC','grade':'B（方向一致但仅 105 期）',
  'value':'IC 0.0434 → 0.0501','source':'reports/data/macro_state_model.json',
  'fails_if':'换训练窗口后不再成立（尚未充分检验）'},
 {'claim':'CTA 与股票策略低相关','grade':'B（2.1 年重叠样本）',
  'value':'corr 0.161','source':'reports/data/combo.json',
  'fails_if':'股灾/流动性危机中相关性会趋近 1（未检验）'},
 {'claim':'美林时钟做择时有效','grade':'D（已被否定）',
  'value':'超额 +2.53% → −1.94%','source':'reports/data/clock_model.json',
  'fails_if':'—'},
 {'claim':'面板无幸存者偏差','grade':'F（已证伪）',
  'value':'缺 123 只退市股（占 2.45%）','source':'reports/data/delisted_check.csv',
  'fails_if':'—'}
]
json.dump(out, open(OUT / 'forecast.json','w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({'state':out['state']['phase'],'clusters':[c['cluster_label']+'('+str(c['size'])+')' for c in clusters],
                  'pred_date':out['prediction']['as_of'],'top3':out['prediction']['top20'][:3]}, ensure_ascii=False, indent=1))
print('DONE')
