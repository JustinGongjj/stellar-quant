"""CTA 趋势策略（经久不衰的那类）：时序动量 + 波动率目标 + 多品种等风险。

规则（经典趋势跟踪，与华泰「真假序列」系列一致）：
  1. 对每个品种算 20/60/120 日时序动量（sum of sign(log return) 或简单收益）；
  2. 信号 = 三个周期动量的符号之和 → ∈[-3,+3]，除以 3 归一；
  3. 仓位 = 信号 × (目标波动 / 该品种近 60 日波动)，每个品种上限 20%；
  4. 组合层面波动率目标 10%，超过则整体缩放；
  5. 成本：单边 2bp（商品期货手续费+滑点），日内不重复计。
对照：买入持有等权商品组合、以及不同周期数的敏感性。
"""
import sys, json, numpy as np, pandas as pd
from stellarq.paths import CACHE as stellarq_cache,  CACHE, DATA  # noqa: E402
OUT = DATA
CACHE = stellarq_cache / 'futures.parquet'
df=pd.read_parquet(CACHE)
df=df.sort_values(['symbol','date'])
# pivot
close=df.pivot_table(index='date', columns='symbol', values='close').sort_index()
ret=close.pct_change()
roll=df.pivot_table(index='date', columns='symbol', values='roll_gap', aggfunc='max').reindex(close.index).fillna(False).astype(bool)
ret=ret.mask(roll)   # 换月跳空不计收益
def strat(periods=(20,60,120), vol_target=0.10, max_w=0.20, cost_bps=2.0):
    vols=ret.rolling(60,min_periods=30).std()*np.sqrt(252)
    sig=sum(np.sign(close/close.shift(p)-1.0) for p in periods)/len(periods)
    raw=sig*(vol_target/vols.clip(lower=0.05))
    w=raw.clip(-max_w,max_w)
    w=w.div(w.abs().sum(axis=1).clip(lower=1e-9),axis=0).mul(1.0)   # 归一到总敞口 1
    gross=(w.shift(1)*ret).sum(axis=1)
    turn=w.diff().abs().sum(axis=1)
    net=gross-turn*cost_bps/1e4
    # 组合波动率目标（用过去 60 日实现波动缩放，滞后一期避免前视）
    realized=net.rolling(60,min_periods=30).std()*np.sqrt(252)
    scale=(vol_target/realized).shift(1).clip(upper=3.0).fillna(0)
    net_s=net*scale
    return net_s, w, {'gross_ann':float(gross.mean()*252),'turnover_daily':float(turn.mean()),
                      'n':int(net_s.notna().sum())}
def stats(net, label):
    n=net.dropna()
    ann=float(n.mean()*252); vol=float(n.std()*np.sqrt(252))
    nav=(1+n).cumprod(); dd=float((nav/nav.cummax()-1).min())
    by_year=n.groupby(n.index.year).apply(lambda x: float((1+x).prod()-1)).to_dict()
    return {'label':label,'ann':ann,'vol':vol,'sharpe':ann/vol if vol>0 else None,'max_dd':dd,
            'calmar':ann/abs(dd) if dd<0 else None,'win_daily':float((n>0).mean()),
            'start':str(n.index.min().date()),'end':str(n.index.max().date()),
            'by_year':{str(k):round(v,4) for k,v in by_year.items()}}
res={}
net,w,meta=strat()
res['trend_20_60_120']=stats(net,'趋势 20/60/120')
res['trend_20_60_120'].update(meta)
for name,per in [('trend_10_30',(10,30)),('trend_60_120_250',(60,120,250)),('trend_120_only',(120,))]:
    n2,_,m2=strat(periods=per)
    res[name]=stats(n2,name); res[name].update(m2)
# 对照：等权买入持有
bh=ret.mean(axis=1)
res['buy_hold_equal']=stats(bh,'等权买入持有')
# 对照：随机信号（打乱符号）
rng=np.random.default_rng(7)
vols=ret.rolling(60,min_periods=30).std()*np.sqrt(252)
randw=pd.DataFrame(rng.choice([-1,1],size=ret.shape),index=ret.index,columns=ret.columns)*(0.10/vols.clip(lower=0.05))
randw=randw.clip(-0.2,0.2); randw=randw.div(randw.abs().sum(axis=1).clip(lower=1e-9),axis=0)
res['random_signal']=stats((randw.shift(1)*ret).sum(axis=1)-randw.diff().abs().sum(axis=1)*2/1e4,'随机信号')
for k,v in res.items():
    print(f"{v['label']:20s} 年化 {v['ann']*100:6.2f}% 波动 {v['vol']*100:5.2f}% Sharpe {v['sharpe']:.2f} 最大回撤 {v['max_dd']*100:6.2f}% Calmar {v['calmar']:.2f}")
json.dump(res, open(OUT / 'cta_trend.json','w'), ensure_ascii=False, indent=1, default=float)
