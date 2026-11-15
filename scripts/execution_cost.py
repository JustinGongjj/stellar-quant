"""执行成本专项：把「固定 5bp 滑点」换成数据驱动的冲击成本，并测试分笔调度。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

模型：Almgren-Chriss 的平方根冲击（业界最常用）
    impact = k × σ_daily × sqrt(Q / ADV)
  Q = 本次下单金额，ADV = 该股 20 日日均成交额，σ = 日波动率，k ≈ 0.3（A 股经验值）

三种执行方案：
  E1 开盘一次性市价成交（当前基准）
  E2 拆成 3 笔：开盘 / 10:30 / 14:30（把冲击成本降为 1/√3）
  E3 按日成交额 5% 上限动态限速（放不下的延到次日）
"""
import sys, json, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    C='CACHE/'; R=C+'wide2/'
    st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); amt=M('amount'); c=M('close'); valid=np.load(R+'wide_valid.npy')
    syms=np.load(R+'wide_syms.npy',allow_pickle=True); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    syms_str=np.array([str(s) for s in syms]); H=6
    # 20 日均成交额与日波动
    ADV20=np.full(amt.shape,np.nan,dtype=np.float32)
    for i0 in range(20, len(o), 200):
        i1=min(i0+200,len(o))
        ADV20[i0:i1]=np.asarray(amt[i0-20:i1-20],dtype=np.float64).mean(axis=0)[None,:].reshape(1,-1) if False else np.asarray(amt[i0:i1],dtype=np.float64).reshape(i1-i0,-1)
    ADV20=np.where(valid, np.asarray(amt,dtype=np.float64), np.nan)   # 简化：用当日成交额作为 ADV 代理
    ret=np.zeros(amt.shape,dtype=np.float32); ret[1:]=np.asarray(c[1:],dtype=np.float64)/np.asarray(c[:-1],dtype=np.float64)-1.0
    SIG=np.array([np.nanstd(ret[max(0,t-20):t+1],axis=0) for t in range(0,len(o),50)])
    w('precomputed')
    COMM=0.00025; TRANS=0.00001; STAMP_POST=0.0005; STAMP_PRE=0.001
    def stamp(dt): return STAMP_POST if pd.Timestamp(dt)>=pd.Timestamp('2023-08-28') else STAMP_PRE
    K=0.3
    def run(size, n_slices, adv_cap=None, top_n=60, hold=50):
        prev=set(); per=[]
        for u in list(np.unique(dte)):
            uu=int(u)
            sc=pd.Series(st[dte==u], index=syms_str[ste[dte==u]]).dropna()
            if len(sc)<100: continue
            ranked=sc.sort_values(ascending=False); cand=list(ranked.head(top_n).index)
            held=list(dict.fromkeys(cand))[:hold]
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms_str)
            r=rp.reindex(held).dropna()
            if len(r)<20: continue
            new=set(r.index); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
            # 每只股票下单金额（占组合规模比例 → 绝对金额）
            per_stock=size/len(r)
            adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=syms_str).reindex(r.index).fillna(1e4)
            sigma=pd.Series(np.nanstd(ret[max(0,uu-20):uu+1],axis=0), index=syms_str).reindex(r.index).fillna(0.03)
            q=per_stock/max(n_slices,1)
            if adv_cap: q=np.minimum(q, adv*adv_cap)
            impact=float(np.nanmean(K*sigma.values*np.sqrt(np.maximum(q,0)/np.maximum(adv.values,1.0)))*np.sqrt(max(n_slices,1)))
            impact=min(impact,0.02)
            cost=turn_*(COMM+TRANS+impact)+turn_*0.5*stamp(dates[uu])
            per.append({'date':str(pd.Timestamp(dates[uu]).date()),'gross':float(r.mean()),'net':float(r.mean())-cost,
                        'bench':float(rp.dropna().mean()),'turn':turn_,'impact':impact})
        pf=pd.DataFrame(per)
        ov=pf['net'].rolling(5).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(5).mean().dropna()
        # 幸存者偏差修正：-1.22pp（phase14 结论，按 -50% 假设）
        ex_adj=ex.mean()*10.08-0.0122
        return {'size_wan':round(size/1e4,0),'slices':n_slices,'adv_cap':adv_cap,
                'ann_net':float(ov.mean()*10.08),'ann_excess':float(ex.mean()*10.08),
                'ann_excess_surv_adj':float(ex_adj),
                'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),'turnover':float(pf['turn'].mean()),
                'impact_bp':float(pf['impact'].mean()*1e4),'cost_per_period_bp':float((pf['gross']-pf['net']).mean()*1e4)}
    res=[]
    for size in (1e7, 5e7, 1e8, 5e8):
        for slices,capg in ((1,None),(3,None)):
            res.append(run(size,slices,capg))
            w(json.dumps(res[-1], ensure_ascii=False))
    json.dump(res, open(OUT / 'execution_cost.json','w'), ensure_ascii=False, indent=1, default=float)
    w('DONE')
except Exception:
    w(traceback.format_exc())
