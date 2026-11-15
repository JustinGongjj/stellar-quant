"""为「未来星图」v2 页面构造完整数据（星座 + 引力井 + 事件脉冲 + 证据闸门）。

在 forecast.json 基础上补充：
  * scenarios   —— 四种情景（复苏/过热/滞胀/衰退）的引力井，参数来自真实历史相位统计与资产表现
  * events      —— 事件脉冲，来自巨潮公告事件因子的真实 IC/ICIR
  * timeline    —— 相位历史时间轴（月频）
"""
import sys, json, glob; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.paths import DATA, ROOT  # noqa: E402
OUT = DATA
base=json.load(open(OUT / 'forecast.json'))

# ---------- 情景引力井：来自真实历史
cl=pd.read_csv(OUT / 'clock_history.csv', parse_dates=['Unnamed: 0'])
ca=json.load(open(OUT / 'clock_assets.json'))
phase_stats=cl['phase'].value_counts().to_dict()
total=sum(v for k,v in phase_stats.items() if k!='无法判定')
SCEN={
 '复苏':{'desc':'增长↑ 通胀↓','assets':'股票 > 债券 > 商品','color':'#4ade80'},
 '过热':{'desc':'增长↑ 通胀↑','assets':'商品 > 股票 > 债券','color':'#f59e0b'},
 '滞胀':{'desc':'增长↓ 通胀↑','assets':'现金/商品抗跌','color':'#ef4444'},
 '衰退':{'desc':'增长↓ 通胀↓','assets':'债券 > 现金 > 股票','color':'#60a5fa'},
}
scenarios=[]
for name,meta in SCEN.items():
    hist=phase_stats.get(name,0)
    prob=hist/total if total else 0
    idx_perf=ca.get('沪深300',{}).get(name,{})
    scenarios.append({
      'name':name,'desc':meta['desc'],'assets':meta['assets'],'color':meta['color'],
      'months':int(hist),'hist_prob':round(float(prob),4),
      'is_current':name==base['state']['phase'],
      'index_ann':float(idx_perf.get('ann',0)) if idx_perf else None,
      'index_sharpe':float(idx_perf.get('sharpe',0)) if idx_perf else None,
      'gravity':round(float(max(0.15, prob*3)),3),
    })
base['scenarios']=scenarios

# ---------- 事件脉冲
ev=pd.read_csv(OUT / 'event_factors.csv')
LABEL={'ev_abnormal_20d':'异常波动','ev_reduce_60d':'减持','ev_increase_60d':'增持',
       'ev_forecast_20d':'业绩预告','ev_express_20d':'业绩快报','ev_buyback_60d':'回购',
       'ev_restructure_120d':'重组','ev_penalty_120d':'处罚'}
pulses=[]
for _,r in ev.iterrows():
    if r['factor'] not in LABEL: continue
    sig=abs(r['t_nw'])
    pulses.append({'name':LABEL[r['factor']],'icir':float(r['icir']),'t':float(r['t_nw']),
                   'q':float(r['q_bh']),'events':int(r['event_days']),
                   'strength':round(float(min(1.0, sig/8.0)),3),
                   'significant':bool(r['q_bh']<0.05)})
pulses.sort(key=lambda x:-x['strength'])
base['events']=pulses

# ---------- 相位时间轴（近 48 个月）
tl=cl[['Unnamed: 0','phase']].tail(48)
base['timeline']=[{'month':str(pd.Timestamp(m).strftime('%Y-%m')),'phase':ph} for m,ph in zip(tl['Unnamed: 0'],tl['phase'])]

json.dump(base, open(OUT / 'forecast.json','w'), ensure_ascii=False, indent=1)
print('scenarios:', [(s['name'], s['hist_prob'], s['is_current']) for s in scenarios])
print('events:', [(p['name'], p['strength'], p['significant']) for p in pulses[:5]])
print('timeline months:', len(base['timeline']))
