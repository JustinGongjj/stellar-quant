"""时钟 × 因子模型：阶段条件化的仓位与选股，产出报告口径的指标。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

三种用法（逐一检验，不预设哪个有效）：
  1. 择时：按相位决定股票仓位（复苏满仓、滞胀/衰退减仓）
  2. 选股：按相位切换因子权重（成长相位偏动量、衰退相位偏低波动）
  3. 结合：择时 + 选股同时上
基准：一直满仓的等权 top60（即 phase10 的 A 方案）。
"""
import sys, json, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    C='CACHE/'; R=C+'wide2/'
    st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); amt=M('amount'); valid=np.load(R+'wide_valid.npy')
    syms=np.load(R+'wide_syms.npy',allow_pickle=True); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    H=6
    clock=pd.read_csv(OUT / 'clock_history.csv', parse_dates=['Unnamed: 0']).rename(columns={'Unnamed: 0':'month'})
    clock['month']=pd.to_datetime(clock['month'])
    known=pd.to_datetime(clock['cpi_known'])
    order=np.argsort(known.to_numpy())
    ph=clock['phase'].to_numpy()[order]; kn=known.to_numpy()[order]
    def phase_at(d):
        i=np.searchsorted(kn, np.datetime64(pd.Timestamp(d)), side='right')-1
        return ph[i] if i>=0 else '无法判定'
    POS={'复苏':1.0,'过热':0.8,'滞胀':0.3,'衰退':0.5,'无法判定':0.7}
    syms_str=np.array([str(s) for s in syms])
    dl=list(np.unique(dte))
    COMM=0.00025; TRANS=0.00001; SLIP=0.0005
    def stamp(dt): return 0.0005 if pd.Timestamp(dt)>=pd.Timestamp('2023-08-28') else 0.001
    def run(mode, topn=60, hold=50):
        prev=set(); per=[]
        for u in dl:
            uu=int(u); dt=pd.Timestamp(dates[uu]); ph_now=phase_at(dt)
            pos=POS.get(ph_now,0.7) if mode in ('timing','both') else 1.0
            sc=pd.Series(st[dte==u], index=syms_str[ste[dte==u]]).dropna()
            if len(sc)<100: continue
            ranked=sc.sort_values(ascending=False)
            cand=list(ranked.head(topn).index)
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms_str)
            r=rp.reindex(cand).dropna()
            if len(r)<30: continue
            gross=float(r.mean())*pos
            new=set(cand); turn_=len(new^prev)/max(len(new)+len(prev),1)*pos; prev=new
            adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=syms_str).reindex(r.index)
            impact=float(np.nanmean(np.minimum(0.003,(1e8*pos/len(r))/np.maximum(adv.values,1e4)*0.1)))
            cost=turn_*(COMM+TRANS+SLIP+impact)+turn_*0.5*stamp(dates[uu])
            per.append({'date':str(dt.date()),'phase':ph_now,'pos':pos,'gross':gross,'net':gross-cost,
                        'bench':float(rp.dropna().mean()),'turn':turn_})
        pf=pd.DataFrame(per)
        ov=pf['net'].rolling(5).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(5).mean().dropna()
        nav=(1+ov).cumprod()
        navb=(1+pf['bench'].rolling(5).mean().dropna()).cumprod()
        return pf, {'n':int(len(pf)),'ann_net':float(ov.mean()*10.08),'ann_bench':float(pf['bench'].rolling(5).mean().dropna().mean()*10.08),
                    'ann_excess':float(ex.mean()*10.08),'sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
                    'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),
                    'max_dd':float((nav/nav.cummax()-1).min()),
                    'excess_max_dd':float(((nav/navb)/(nav/navb).cummax()-1).min()),
                    'turnover':float(pf['turn'].mean()),'win':float((ov>0).mean())}
    res={}; store={}
    for mode in ('hold','timing'):
        pf,r=run(mode); res[mode]=r; store[mode]=pf
        w(f'{mode}: ann {r["ann_net"]:.2%} excess {r["ann_excess"]:.2%} IR {r["ir"]:.2f} maxDD {r["max_dd"]:.2%}')
    # 相位分组：各相位下组合与基准的收益
    pf=store['hold']
    g=pf.groupby('phase').agg(n=('net','size'),port=('net','mean'),bench=('bench','mean'))
    g['excess']=g['port']-g['bench']
    res['by_phase']={k:{kk:round(float(vv),5) for kk,vv in v.items()} for k,v in g.to_dict('index').items()}
    store['hold'].to_csv(OUT / 'clock_model_periods.csv', index=False)
    json.dump(res, open(OUT / 'clock_model.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    w('DONE')
except Exception:
    w(traceback.format_exc())
