"""组合优化（修正版）：用「解析投影 + 权重归一」而非梯度下降。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

上一版的问题：
  1. 梯度步长与惩罚项让权重塌缩成等权（持仓恒为 60）；
  2. 样本期与基准不一致（74 vs 105）。
本版：
  * 先把候选集限定为 top-N（默认 60），权重用软最大化（score 的指数化）分配；
  * 个股上限用「水填充」投影；
  * 行业上限用迭代再分配；
  * 三档在**完全相同的候选集与日期**上比较。
"""
import sys, glob, json, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    C='CACHE/'; R=C+'wide2/'
    st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); amt=M('amount'); cap=M('float_mktcap'); vol=M('volume'); valid=np.load(R+'wide_valid.npy')
    syms=np.load(R+'wide_syms.npy',allow_pickle=True); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    H=6
    ind=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(C+'industry/*.parquet'))], ignore_index=True).dropna(subset=['BOARD_NAME'])
    ind_latest=ind.drop_duplicates('symbol',keep='last').set_index('symbol')['BOARD_NAME'].to_dict()
    syms_str=np.array([str(s) for s in syms])
    ind_map=np.array([ind_latest.get(s,'UNKNOWN') for s in syms_str])
    w(f'industry unknown {int((ind_map=="UNKNOWN").sum())}/{len(syms)}')
    COMM=0.00025; TRANS=0.00001; SLIP=0.0005
    def stamp(dt): return 0.0005 if pd.Timestamp(dt)>=pd.Timestamp('2023-08-28') else 0.001
    dl=list(np.unique(dte))
    sc_by_date={u: pd.Series(st[dte==u], index=syms_str[ste[dte==u]]) for u in dl}
    prob=valid.mean(axis=1)
    keep_dates=[u for u in dl if prob[u]>=0.5]
    w(f'kept {len(keep_dates)}/{len(dl)} dates')
    def waterfill(raw, cap_w):
        """把非负权重投影到 {Σw=1, 0≤w≤cap_w}（水填充）。"""
        n=len(raw); wv=np.zeros(n)
        if n*cap_w<1: cap_w=1.0/n
        order=np.argsort(-raw); rem=1.0; free=n
        for i in order:
            if free<=0: break
            share=min(raw[i], rem/free) if raw[i]>0 else rem/free
            share=min(share, cap_w)
            wv[i]=share; rem-=share; free-=1
            if rem<=1e-12: break
        s=wv.sum()
        return wv/s if s>0 else np.full(n,1.0/n)
    def apply_ind_cap(wv, inds, ind_cap, iters=30):
        for _ in range(iters):
            excess=0.0
            for u in np.unique(inds):
                m=inds==u; tot=wv[m].sum()
                if tot>ind_cap+1e-12:
                    excess+=tot-ind_cap; wv[m]*=ind_cap/tot
            if excess<=1e-9: break
            room=(wv<0.999*1.0/len(wv)) 
            under=np.array([wv[inds==u].sum()<ind_cap-1e-9 for u in np.unique(inds)])
            ok_mask=np.zeros(len(wv),bool)
            for u,can in zip(np.unique(inds),under):
                if can: ok_mask|= (inds==u)
            if not ok_mask.any(): break
            wv[ok_mask]+=excess*room[ok_mask]/max(room[ok_mask].sum(),1)
        return wv/wv.sum()
    def backtest(mode, topn=60, hold=50, ind_cap=1.0, cap_w=1.0):
        prev=set(); per=[]
        for u in keep_dates:
            uu=int(u); sc=sc_by_date[u].dropna()
            if len(sc)<100: continue
            ranked=sc.sort_values(ascending=False)
            cand=list(ranked.head(topn).index)
            if len(cand)<30: continue
            s=sc.reindex(cand).to_numpy(dtype=np.float64)
            s=(s-np.nanmean(s))/(np.nanstd(s)+1e-12)
            z=(s-s.min())/(s.max()-s.min()+1e-12)+0.05       # 保底权重，避免全押一只
            if mode=='A':
                wts={c:1.0/len(cand) for c in cand}
            else:
                raw=np.exp(1.5*z)
                wv=waterfill(raw, cap_w)
                if ind_cap<1.0:
                    wv=apply_ind_cap(wv, ind_map[np.array([list(syms_str).index(c) for c in cand])], ind_cap)
                else:
                    wv=wv/wv.sum()
                keep_n=min(hold,len(cand))
                order=np.argsort(-wv)[:keep_n]
                wv=np.zeros(len(cand)); wv[order]=1.0/keep_n if mode=='A' else None
                if mode=='B': wv[order]=np.array([1.0 for _ in order])/keep_n
                wv=np.zeros(len(cand))
                base=waterfill(raw, cap_w)
                if ind_cap<1.0:
                    base=apply_ind_cap(base, ind_map[np.array([list(syms_str).index(c) for c in cand])], ind_cap)
                sel=np.argsort(-base)[:keep_n]
                wv=np.zeros(len(cand)); wv[sel]=base[sel]; wv=wv/wv.sum()
                wts={cand[i]:float(wv[i]) for i in range(len(cand))}
            held=list(wts.keys())
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms_str)
            r=rp.reindex(held); wv=pd.Series(wts); ok=r.notna()
            if ok.sum()<len(wv)*0.8: continue
            wv=wv[ok]; wv=wv/wv.sum(); r=r[ok]
            gross=float((r*wv).sum())
            new=set(wv.index); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
            adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=syms_str).reindex(wv.index)
            impact=float(np.nanmean(np.minimum(0.003,(1e8*wv.values)/np.maximum(adv.values,1e4)*0.1)))
            cost=turn_*(COMM+TRANS+SLIP+impact)+turn_*0.5*stamp(dates[uu])
            per.append({'date':str(pd.Timestamp(dates[uu]).date()),'gross':gross,'net':gross-cost,
                        'bench':float(rp.dropna().mean()),'turn':turn_,'n':len(wv),
                        'hhi':float((wv**2).sum())})
        pf=pd.DataFrame(per)
        if not len(pf): return None
        ov=pf['net'].rolling(5).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(5).mean().dropna()
        nav=(1+ov).cumprod()
        r={'n':int(len(pf)),'ann_net':float(ov.mean()*10.08),'ann_excess':float(ex.mean()*10.08),
           'sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
           'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),
           'turnover':float(pf['turn'].mean()),'max_dd':float((nav/nav.cummax()-1).min()),
           'hhi':float(pf['hhi'].mean())}
        pf.to_csv(OUT+f'opt2_{mode}.csv',index=False)
        return r
    res={'A_等权60':backtest('A'),'B_IC加权+上限3%':backtest('B',cap_w=0.03),
         'C_IC加权+行业20%+上限3%':backtest('C',cap_w=0.03,ind_cap=0.20)}
    json.dump(res, open(OUT / 'portfolio_opt2.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    w('DONE')
except Exception:
    w(traceback.format_exc())
