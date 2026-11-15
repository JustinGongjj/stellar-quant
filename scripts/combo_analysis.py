"""CTA 与股票策略的相关性与组合检验。

股票策略：v4 得分 top60，5 日调仓（沿用 execution 脚本的口径），2024-07 → 2026-09
CTA：趋势 10/30，日频 → 按股票策略的调仓日重采样

组合：固定 60/40、70/30 等权，以及"按滚动 60 日协方差的最优权重（风险平价）"
"""
import sys, json, numpy as np, pandas as pd
from stellarq.paths import CACHE, DATA  # noqa: E402
OUT = DATA
C='CACHE/'; R=C+'wide2/'
# ---- 股票策略逐期净收益
st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
o=np.load(R+'wide_open.npy', mmap_mode='r'); amt=np.load(R+'wide_amount.npy', mmap_mode='r')
valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
H=6; K=0.0137; COMM=0.00025; TRANS=0.00001
prev=set(); rows=[]
for u in list(np.unique(dte)):
    uu=int(u)
    sc=pd.Series(st[dte==u], index=syms[ste[dte==u]]).dropna()
    if len(sc)<100: continue
    cand=list(sc.sort_values(ascending=False).head(60).index)
    rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
    rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms)
    r=rp.reindex(cand).dropna()
    if len(r)<40: continue
    new=set(r.index); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
    adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=syms).reindex(r.index).fillna(1e4)
    impact=float(np.nanmean(K*np.sqrt((1e8/len(r))/np.maximum(adv.values,1.0))))
    cost=turn_*(COMM+TRANS+impact)+turn_*0.5*0.0005
    rows.append({'date':pd.Timestamp(dates[uu]),'stock':float(r.mean())-cost})
eq=pd.DataFrame(rows).set_index('date').sort_index()
print('股票策略期数', len(eq), eq.index.min().date(), eq.index.max().date())
# ---- CTA 逐日净收益（趋势 10/30，同上一轮参数）
fut=pd.read_parquet(C+'futures.parquet').sort_values(['symbol','date'])
close=fut.pivot_table(index='date',columns='symbol',values='close').sort_index()
ret=close.pct_change()
roll=fut.pivot_table(index='date',columns='symbol',values='roll_gap',aggfunc='max').reindex(close.index).fillna(False).astype(bool)
ret=ret.mask(roll)
vols=ret.rolling(60,min_periods=30).std()*np.sqrt(252)
sig=sum(np.sign(close/close.shift(p)-1.0) for p in (10,30))/2
w=(sig*(0.10/vols.clip(lower=0.05))).clip(-0.2,0.2)
w=w.div(w.abs().sum(axis=1).clip(lower=1e-9),axis=0)
gross=(w.shift(1)*ret).sum(axis=1); turn=w.diff().abs().sum(axis=1)
cta=(gross-turn*2/1e4)
real=cta.rolling(60,min_periods=30).std()*np.sqrt(252)
cta=cta*(0.10/real).shift(1).clip(upper=3.0).fillna(0)
# 把 CTA 日频收益按股票策略的持有期（5 个交易日）聚合
cta5=cta.groupby(pd.Series(cta.index)).sum()  # 日频
# 与股票策略对齐：股票期收益 = T+1 开盘 → T+6 开盘（5 个交易日）。
# CTA 用同一窗口（调仓日之后第 1 至第 5 个交易日）的复利收益，而不是简单的 5 日累加，
# 且必须与股票策略「同一套调仓日」——之前的实现取的是 d0 之后前 5 个交易日，口径是对的，
# 但累加方式（sum）在日波动 0.7% 时会系统性低估，改用复利。
cta_period=[]
for d0 in eq.index:
    win=cta.loc[(cta.index>d0)]
    if len(win)<5: continue
    cta_period.append({'date':d0,'cta':float(np.prod(1.0+win.iloc[:5].to_numpy())-1.0)})
ct=pd.DataFrame(cta_period).set_index('date')
df=eq.join(ct,how='inner').dropna()
print('对齐后', len(df), '期；CTA 期均', round(float(df['cta'].mean())*100,3), '%（×10.08 →', round(float(df['cta'].mean())*10.08*100,2), '% 年化）')
corr=float(df['stock'].corr(df['cta']))
print(f'相关性 {corr:.3f}')
def stats(x,label):
    """注意：这里 x 是「每 5 个交易日一期的收益」（股票策略是一期，CTA 是 5 日累加）。
    重叠修正：5 期移动平均后每 25 个交易日算一个独立观测 → 年化系数 = 252/25 = 10.08。
    这是 v3 阶段踩过的坑，务必不要再乘 50.4。"""
    x=np.asarray(x)
    # 关键：股票策略「每周调仓、持有 5 日」，相邻两期共享 4/5 的价格路径，
    # 因此每 5 个期才算 1 个独立观测 → 独立观测数 = len(x)/5，年化系数 = 252/(25)。
    # 若按 len(x)*5 个交易日直接复利，会把重叠重复计入，年化被放大 5 倍（实测 6.25% → 35.40%）。
    n_indep=len(x)/5.0
    ann=(np.prod(1.0+x))**(10.08/n_indep)-1.0
    vol=x.std(ddof=1)/5.0*np.sqrt(252.0)
    nav=np.cumprod(1+x); dd=float((nav/np.maximum.accumulate(nav)-1).min())
    return {'label':label,'ann':float(ann),'vol':float(vol),'sharpe':float(ann/vol) if vol>0 else None,
            'max_dd':dd,'win':float((x>0).mean())}
res={'corr':corr,'n':int(len(df)),
     'stock':stats(df['stock'],'股票策略'),'cta':stats(df['cta'],'CTA 趋势')}
for wgt in (0.3,0.4,0.5):
    res[f'combo_{int(wgt*100)}_{int((1-wgt)*100)}']=stats(wgt*df['stock']+(1-wgt)*df['cta'],f'股票{int(wgt*100)}/CTA{int((1-wgt)*100)}')
# 风险平价（滚动 60 期）
vol_s=df['stock'].rolling(20,min_periods=10).std(); vol_c=df['cta'].rolling(20,min_periods=10).std()
w_s=(1/vol_s)/((1/vol_s)+(1/vol_c))
rp=(w_s*df['stock']+(1-w_s)*df['cta']).dropna()
res['risk_parity']=stats(rp,'风险平价')
for k,v in res.items():
    if isinstance(v,dict):
        print(f"{v['label']:16s} 年化 {v['ann']*100:6.2f}% 波动 {v['vol']*100:5.2f}% Sharpe {v['sharpe']:.2f} 回撤 {v['max_dd']*100:6.2f}%")
json.dump(res, open(OUT / 'combo.json','w'), ensure_ascii=False, indent=1, default=float)
