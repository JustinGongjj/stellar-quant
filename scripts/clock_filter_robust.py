"""F1 过滤器的稳健性检验：换阈值、换相位定义、分年拆解，看是否只是运气。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

检验项：
  R1 只对「滞胀」过滤（现状） vs 只对「滞胀+衰退」 vs 只对「无法判定」
  R2 相位判定 lookback 从 3 改成 6 个月（判据更平滑）
  R3 分年拆解，看收益提升集中在哪一年
"""
import sys, json, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    from stellarq.macro.clock import macro_panel, classify, PHASES
    C='CACHE/'; R=C+'wide2/'
    st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); amt=M('amount'); valid=np.load(R+'wide_valid.npy')
    syms=np.load(R+'wide_syms.npy',allow_pickle=True); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    syms_str=np.array([str(s) for s in syms]); H=6
    panel=macro_panel('2012-01-01')
    def make_phase(lookback):
        cl=classify(panel, lookback=lookback)
        kn=pd.to_datetime(cl['cpi_known']).to_numpy(); ph=cl['phase'].to_numpy()
        order=np.argsort(kn); kn=kn[order]; ph=ph[order]
        def at(d):
            i=np.searchsorted(kn, np.datetime64(pd.Timestamp(d)), side='right')-1
            return ph[i] if i>=0 else '无法判定'
        return at
    COMM=0.00025; TRANS=0.00001; SLIP=0.0005
    def stamp(dt): return 0.0005 if pd.Timestamp(dt)>=pd.Timestamp('2023-08-28') else 0.001
    dl=list(np.unique(dte))
    def run(phase_at, block_set, topn=60, hold=50):
        prev=set(); per=[]
        for u in dl:
            uu=int(u); ph=phase_at(dates[uu])
            sc=pd.Series(st[dte==u], index=syms_str[ste[dte==u]]).dropna()
            if len(sc)<100: continue
            ranked=sc.sort_values(ascending=False); cand=list(ranked.head(topn).index)
            pick=[x for x in prev if x in sc.index][:hold] if ph in block_set else cand
            held=list(dict.fromkeys(pick))[:hold]
            if len(held)<20: held=cand[:hold]
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms_str)
            r=rp.reindex(held).dropna()
            if len(r)<20: continue
            new=set(r.index); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
            adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=syms_str).reindex(r.index)
            impact=float(np.nanmean(np.minimum(0.003,(1e8/len(r))/np.maximum(adv.values,1e4)*0.1)))
            cost=turn_*(COMM+TRANS+SLIP+impact)+turn_*0.5*stamp(dates[uu])
            per.append({'date':str(pd.Timestamp(dates[uu]).date()),'net':float(r.mean())-cost,'bench':float(rp.dropna().mean()),'turn':turn_})
        pf=pd.DataFrame(per)
        ov=pf['net'].rolling(5).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(5).mean().dropna()
        by_year=pf.assign(y=pf['date'].str[:4]).groupby('y').apply(lambda d: float((d['net']-d['bench']).sum())).to_dict()
        return {'n':int(len(pf)),'ann_net':float(ov.mean()*10.08),'ann_excess':float(ex.mean()*10.08),
                'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),'turnover':float(pf['turn'].mean()),
                'excess_by_year':{k:round(v,4) for k,v in by_year.items()}}
    res={}
    at3=make_phase(3); at6=make_phase(6)
    res['baseline_lb3']=run(at3,set())
    res['F1_滞胀_lb3']=run(at3,{'滞胀'})
    res['F1_滞胀+衰退_lb3']=run(at3,{'滞胀','衰退'})
    res['F1_无法判定_lb3']=run(at3,{'无法判定'})
    res['baseline_lb6']=run(at6,set())
    res['F1_滞胀_lb6']=run(at6,{'滞胀'})
    for k,v in res.items():
        w(f'{k:22s} 超额 {v["ann_excess"]:+.2%} IR {v["ir"]:+.2f} 换手 {v["turnover"]:.2f} 分年 {v["excess_by_year"]}')
    json.dump(res, open(OUT / 'clock_filter_robust.json','w'), ensure_ascii=False, indent=1, default=float)
    w('DONE')
except Exception:
    w(traceback.format_exc())
