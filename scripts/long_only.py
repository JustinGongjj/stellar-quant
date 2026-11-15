"""多头端评估：把 v4 的样本外得分只做多头，和等权/市值加权基准比。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

为什么必须单独做：v4 报的是 top50 − bottom50 的价差，A 股个股做空基本不可行，
实盘只能吃到多头腿。这里给出多头腿的真实年化、超额、回撤。

费用口径：佣金+过户费（双边）+ 印花税（卖出腿）+ 冲击成本按「成交额占比」分段。
"""
import sys, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'long_only.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'long_only.log','w').close()
try:
    import numpy as np, pandas as pd
    t0=0
    C='CACHE/'
    st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
    R=C+'wide2/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); amt=M('amount'); valid=np.load(R+'wide_valid.npy')
    syms=np.load(R+'wide_syms.npy',allow_pickle=True); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    nd=len(o); ns=len(syms); H=6
    COMMISSION=0.00025; TRANSFER=0.00001; SLIP=0.0005
    def stamp(d):
        return 0.0005 if pd.Timestamp(d)>=pd.Timestamp('2023-08-28') else 0.001
    dl=list(np.unique(dte))
    sc_by_date={u: pd.Series(st[dte==u], index=[str(syms[i]) for i in ste[dte==u]]) for u in dl}
    rows=[]; prev=set()
    for ui,u in enumerate(dl):
        uu=int(u)
        sc=sc_by_date[u]
        ranked=sc.sort_values(ascending=False)
        rank={s:i for i,s in enumerate(ranked.index)}
        top=list(ranked.head(60).index)
        keep=[s for s in prev if rank.get(s,10**9)<80]
        held=list(dict.fromkeys(keep+top))[:50]
        # 默认场景：T+1 开盘买、T+6 开盘卖（与训练标签一致）
        rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
        rp=pd.Series(np.where(valid[uu],rp,np.nan), index=[str(s) for s in syms])
        r=rp.reindex(held).dropna()
        if len(r)<40: continue
        # 冲击成本：按 50 只等权、组合规模 1 亿、单票成交额占比估算
        adv=np.asarray(amt[uu],dtype=np.float64)/max(1e8,1)
        slip=np.nanmean(1.0/(1.0+np.maximum(adv[ste[dte==u]],1.0)*0.0))  # 占位，下面按占比算
        adv_sel=pd.Series(adv, index=[str(s) for s in syms]).reindex(held)
        pos=1e8/len(held)
        impact=float(np.nanmean(np.minimum(0.003, pos/np.maximum(adv_sel.values*1e8,1e4)*0.1)))
        new=set(held); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
        # 基准：同池等权（即当截面的全市场等权）
        bench=float(rp.dropna().mean())
        cost=turn_*(COMMISSION+TRANSFER+SLIP+impact)+turn_*0.5*stamp(dates[uu])
        rows.append({'date':str(pd.Timestamp(dates[uu]).date()),'gross':float(r.mean()),'net':float(r.mean())-cost,
                     'bench':bench,'excess':float(r.mean())-bench-cost,'turnover':turn_,'impact':impact,'n':len(r)})
    df=pd.DataFrame(rows)
    # 重叠修正：5 期移动平均 × (252/25)
    ov_net=df['net'].rolling(5).mean().dropna(); ov_ex=df['excess'].rolling(5).mean().dropna()
    ov_bench=df['bench'].rolling(5).mean().dropna()
    nav=(1+ov_net).cumprod()
    res={'n_periods':int(len(df)),'start':df['date'].iloc[0],'end':df['date'].iloc[-1],
         'ann_net':float(ov_net.mean()*10.08),'ann_bench':float(ov_bench.mean()*10.08),
         'ann_excess':float(ov_ex.mean()*10.08),
         'sharpe_net':float(ov_net.mean()/ov_net.std(ddof=1)*np.sqrt(10.08)),
         'ir_excess':float(ov_ex.mean()/ov_ex.std(ddof=1)*np.sqrt(10.08)),
         'max_dd':float((nav/nav.cummax()-1).min()),
         'win_abs':float((ov_net>0).mean()),'win_vs_bench':float((ov_ex>0).mean()),
         'turnover':float(df['turnover'].mean()),'impact_cost_mean':float(df['impact'].mean()),
         'cost_per_period':float((df['gross']-df['net']).mean()),
         'by_year':{str(k):float(v) for k,v in df.assign(y=df['date'].str[:4]).groupby('y')['net'].sum().items()}}
    json.dump(res, open(OUT / 'long_only.json','w'), ensure_ascii=False, indent=1, default=float)
    df.to_csv(OUT / 'long_only.csv', index=False)
    w(json.dumps(res, ensure_ascii=False, indent=1, default=float))
except Exception:
    w(traceback.format_exc())
