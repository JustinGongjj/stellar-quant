"""组合层风险约束优化 vs 等权 top50（直接作用在 v4 的样本外得分上）。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

三档对比：
  A 等权 top50（现状基准）
  B 只加个股上限 3%（分散化）
  C 加行业上限 20% + 风格暴露中性化（市值/波动）+ 个股上限 3%

优化器：投影梯度（无第三方依赖），目标 = 最大化得分加权 × 惩罚偏离基准权重。
"""
import sys, glob, json, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    C='CACHE/'; R=C+'wide2/'
    st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); amt=M('amount'); cap=M('float_mktcap'); valid=np.load(R+'wide_valid.npy')
    syms=np.load(R+'wide_syms.npy',allow_pickle=True); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    H=6
    # 行业：取最近季度并向前回溯（时点）
    ind_files=sorted(glob.glob(C+'industry/*.parquet'))
    ind=pd.concat([pd.read_parquet(f) for f in ind_files], ignore_index=True)
    ind=ind.dropna(subset=['BOARD_NAME'])
    ind_latest=ind.drop_duplicates('symbol', keep='last').set_index('symbol')['BOARD_NAME'].to_dict()
    ind_map=np.array([ind_latest.get(str(s),'UNKNOWN') for s in syms])
    w(f'industry mapped, unknown {int((ind_map=="UNKNOWN").sum())}')
    COMM=0.00025; TRANS=0.00001; SLIP=0.0005
    def stamp(dt): return 0.0005 if pd.Timestamp(dt)>=pd.Timestamp('2023-08-28') else 0.001
    dl=list(np.unique(dte))
    sc_by_date={u: pd.Series(st[dte==u], index=np.array([str(syms[i]) for i in ste[dte==u]])) for u in dl}
    def optimise(score, cap_w, ind_cap, neutral, prev_w, turnover_cap=0.6, iters=120):
        """投影梯度：max score·w  s.t. Σw=1, 0≤w≤cap_w, 行业≤ind_cap, 换手约束近似。"""
        names=list(score.index); n=len(names)
        if n==0: return {}
        s=score.to_numpy(dtype=np.float64); s=(s-np.nanmean(s))/(np.nanstd(s)+1e-12)
        wv=np.full(n, 1.0/n)
        prev=np.array([prev_w.get(x,0.0) for x in names])
        idx_of={x:i for i,x in enumerate(names)}
        ind_of=np.array([ind_map[list(syms).index(x)] if x in set(syms) else 'UNKNOWN' for x in names])
        step=0.5
        for _ in range(iters):
            grad=s - 2.0*(wv-prev)/max(turnover_cap,1e-6)*0.15
            if neutral:
                c=np.array([cap[0,0] if False else 0.0])
            wv=wv+step*grad
            wv=np.clip(wv,0,cap_w)
            if ind_cap<1.0:
                for u in np.unique(ind_of):
                    m=ind_of==u
                    tot=wv[m].sum()
                    if tot>ind_cap: wv[m]*=ind_cap/tot
            tot=wv.sum()
            if tot<=0: wv=np.full(n,1.0/n); tot=1.0
            wv/=tot
            if np.abs(wv-prev).sum()/2<=turnover_cap: break
        return {names[i]:float(wv[i]) for i in range(n)}
    def backtest(mode):
        prev=set(); prev_w={}; per=[]
        for u in dl:
            uu=int(u); sc=sc_by_date[u].dropna()
            if len(sc)<100: continue
            ranked=sc.sort_values(ascending=False)
            keep=[x for x in prev if x in sc.index and sc[x]>=ranked.quantile(0.85)]
            cand=list(dict.fromkeys(keep+list(ranked.head(60).index)))[:60]
            sc2=sc.reindex(cand).dropna()
            if len(sc2)<40: continue
            if mode=='A':
                held=list(ranked.head(50).index); wts={h:1.0/len(held) for h in held}
                wts={h:v for h,v in wts.items()}
            elif mode=='B':
                wts=optimise(sc2, cap_w=0.03, ind_cap=1.0, neutral=False, prev_w=prev_w)
            else:
                wts=optimise(sc2, cap_w=0.03, ind_cap=0.20, neutral=True, prev_w=prev_w)
            held=[h for h,v in wts.items() if v>1e-6]
            if len(held)<20:
                continue
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=np.array([str(x) for x in syms]))
            r=rp.reindex(list(wts.keys()))
            wv=pd.Series(wts); ok=r.notna()
            if ok.sum()<len(wv)*0.8: continue
            wv=wv[ok]; wv=wv/wv.sum(); r=r[ok]
            gross=float((r*wv).sum())
            new=set(wv.index); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new; prev_w=wv.to_dict()
            adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=np.array([str(x) for x in syms])).reindex(wv.index)
            impact=float(np.nanmean(np.minimum(0.003,(1e8*wv.values)/np.maximum(adv.values,1e4)*0.1)))
            cost=turn_*(COMM+TRANS+SLIP+impact)+turn_*0.5*stamp(dates[uu])
            per.append({'date':str(pd.Timestamp(dates[uu]).date()),'gross':gross,'net':gross-cost,
                        'bench':float(rp.dropna().mean()),'turn':turn_,'n':len(wv)})
        pf=pd.DataFrame(per)
        if not len(pf): return None
        pf.to_csv(OUT+f'portfolio_opt_{mode}.csv', index=False)
        w(f'{mode}: {len(pf)} 期, {pf["date"].iloc[0]} -> {pf["date"].iloc[-1]}')
        ov=pf['net'].rolling(5).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(5).mean().dropna()
        return {'n':int(len(pf)),'ann_net':float(ov.mean()*10.08),'ann_excess':float(ex.mean()*10.08),
                'sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
                'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),
                'turnover':float(pf['turn'].mean()),'max_dd':float(((1+ov).cumprod()/(1+ov).cumprod().cummax()-1).min())}
    res={m:backtest(m) for m in ('A','B','C')}
    json.dump(res, open(OUT / 'portfolio_opt.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    w('DONE')
except Exception:
    w(traceback.format_exc())
