"""时钟当「过滤器」而非「调仓位」：只影响新开仓，不影响已有持仓。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

三种过滤器，逐一检验（不预设哪种有效）：
  F1 滞胀期不新开仓（持仓照常持有到到期）
  F2 滞胀期新开仓只用最高分的一半（缩小新仓）
  F3 滞胀期把新开仓换成低波动/大市值子集（换风格而非空仓）
对照：无过滤（baseline）
"""
import sys, json, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    C='CACHE/'; R=C+'wide2/'
    st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); amt=M('amount'); cap=M('float_mktcap'); vol=M('volume')
    valid=np.load(R+'wide_valid.npy'); syms=np.load(R+'wide_syms.npy',allow_pickle=True); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    H=6
    clock=pd.read_csv(OUT / 'clock_history.csv', parse_dates=['Unnamed: 0'])
    kn=pd.to_datetime(clock['cpi_known']).to_numpy(); ph=clock['phase'].to_numpy()
    order=np.argsort(kn); kn=kn[order]; ph=ph[order]
    def phase_at(d):
        i=np.searchsorted(kn, np.datetime64(pd.Timestamp(d)), side='right')-1
        return ph[i] if i>=0 else '无法判定'
    syms_str=np.array([str(s) for s in syms])
    dl=list(np.unique(dte))
    COMM=0.00025; TRANS=0.00001; SLIP=0.0005
    def stamp(dt): return 0.0005 if pd.Timestamp(dt)>=pd.Timestamp('2023-08-28') else 0.001
    # 预先算好每期可选股票池与得分
    def run(mode, topn=60, hold=50):
        prev=set(); per=[]
        for u in dl:
            uu=int(u); phase=phase_at(dates[uu])
            sc=pd.Series(st[dte==u], index=syms_str[ste[dte==u]]).dropna()
            if len(sc)<100: continue
            ranked=sc.sort_values(ascending=False)
            cand=list(ranked.head(topn).index)
            # 过滤器只影响「本期新进名单」
            if phase=='滞胀':
                if mode=='F1':
                    new_pick=[x for x in prev if x in sc.index][:hold]   # 不新开，只保留旧仓
                elif mode=='F2':
                    new_pick=cand[:max(hold//2,10)]
                elif mode=='F3':
                    # 换成候选池里市值最大的前 20 只（低波动/大盘偏好）
                    sub=pd.Series(np.asarray(cap[uu],dtype=np.float64), index=syms_str).reindex(cand).dropna()
                    new_pick=list(sub.sort_values(ascending=False).head(hold).index)
                else:
                    new_pick=cand
            else:
                new_pick=cand
            held=list(dict.fromkeys(new_pick))[:hold]
            if len(held)<20: held=cand[:hold]
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms_str)
            r=rp.reindex(held).dropna()
            if len(r)<20: continue
            gross=float(r.mean())
            new=set(r.index); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
            adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=syms_str).reindex(r.index)
            impact=float(np.nanmean(np.minimum(0.003,(1e8/len(r))/np.maximum(adv.values,1e4)*0.1)))
            cost=turn_*(COMM+TRANS+SLIP+impact)+turn_*0.5*stamp(dates[uu])
            per.append({'date':str(pd.Timestamp(dates[uu]).date()),'phase':phase,'gross':gross,'net':gross-cost,
                        'bench':float(rp.dropna().mean()),'turn':turn_,'n':len(r)})
        pf=pd.DataFrame(per)
        ov=pf['net'].rolling(5).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(5).mean().dropna()
        nav=(1+ov).cumprod()
        return {'n':int(len(pf)),'ann_net':float(ov.mean()*10.08),'ann_excess':float(ex.mean()*10.08),
                'sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
                'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),
                'max_dd':float((nav/nav.cummax()-1).min()),
                'turnover':float(pf['turn'].mean()),
                'stagnation_net':float(pf.loc[pf['phase']=='滞胀','net'].mean()) if (pf['phase']=='滞胀').any() else None}
    res={}
    for m in ('baseline','F1','F2','F3'):
        res[m]=run(m); w(f'{m}: {json.dumps({k:round(v,4) for k,v in res[m].items() if isinstance(v,float)}, ensure_ascii=False)}')
    json.dump(res, open(OUT / 'clock_filter.json','w'), ensure_ascii=False, indent=1, default=float)
    w('DONE')
except Exception:
    w(traceback.format_exc())
