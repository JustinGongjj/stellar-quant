"""换手 × 容量的权衡曲线：找「超额/容量」的最优点。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

不同持有期 h（5/10/20/40 日）对应不同的标签与换手；
每个持有期再配 3 种组合规模（1/5/20 亿），算校准后的冲击成本与幸存者修正后的超额。

为了不重训模型（太慢），用两个近似：
  * 持有期 h 的得分沿用 v4 模型（训练标签是 6 日），但按 h 期持有、h 期调仓；
  * 标签换成 h 期，IC 会随 h 变化（单独报）。
这能给出**换手-容量**的形状，但不能替代重训。
"""
import sys, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
try:
    import numpy as np, pandas as pd
    C='CACHE/'; R=C+'wide2/'
    st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o=M('open'); amt=M('amount'); valid=np.load(R+'wide_valid.npy')
    syms_str=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
    nd=len(o)
    dl=sorted(set(dte.tolist()))
    COMM=0.00025; TRANS=0.00001; K=0.0137
    def stamp(dt): return 0.0005 if pd.Timestamp(dt)>=pd.Timestamp('2023-08-28') else 0.001
    def run(h, size, top_n=60, hold=50, rebal=None):
        rebal=rebal or h
        dates_by_idx={int(u):i for i,u in enumerate(dl)}
        prev=set(); per=[]; last=None
        for u in dl:
            uu=int(u)
            if last is not None and (uu-last)<rebal: continue
            if uu+1+h>=nd: break
            last=uu
            sc=pd.Series(st[dte==u], index=syms_str[ste[dte==u]]).dropna()
            if len(sc)<100: continue
            cand=list(sc.sort_values(ascending=False).head(top_n).index)
            held=list(dict.fromkeys(cand))[:hold]
            rp=np.asarray(o[uu+1+h],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms_str)
            r=rp.reindex(held).dropna()
            if len(r)<20: continue
            new=set(r.index); turn_=len(new^prev)/max(len(new)+len(prev),1); prev=new
            adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=syms_str).reindex(r.index).fillna(1e4)
            q=size/len(r)
            impact=float(np.nanmean(K*np.sqrt(q/np.maximum(adv.values,1.0))))
            cost=turn_*(COMM+TRANS+impact)+turn_*0.5*stamp(pd.Timestamp(np.datetime64('2015-01-01'))) if False else turn_*(COMM+TRANS+impact)+turn_*0.5*0.00075
            per.append({'gross':float(r.mean()),'net':float(r.mean())-cost,'bench':float(rp.dropna().mean()),
                        'turn':turn_,'impact':impact,'h':h})
        pf=pd.DataFrame(per)
        if len(pf)<10: return None
        # 重叠修正：持有 h 日、每 h 日调仓 → h/h=1 无重叠；但「调仓间隔 h、持有 h」时
        # 相邻期不共享价格路径，年化系数 = 252/h。这里统一用与前面几轮相同的口径：
        # 先用 h 期移动平均压掉重叠，再乘 (252/h)。
        ov=pf['net'].rolling(h).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(h).mean().dropna()
        per_year=252/(h*h)   # h 期平均后的有效独立性：每 h 期算一个等效观测，再乘 252/h
        return {'h':h,'size_yi':round(size/1e8,1),'n':int(len(pf)),
                'ann_net':float(ov.mean()*per_year),'ann_excess':float(ex.mean()*per_year),
                'ann_excess_surv_adj':float(ex.mean()*per_year-0.0122),
                'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(per_year)),
                'turnover':float(pf['turn'].mean()),'impact_bp':float(pf['impact'].mean()*1e4),
                'cost_bp':float((pf['gross']-pf['net']).mean()*1e4)}
    res=[]
    for h in (5,10,20,40):
        for size in (1e8,5e8,2e9):
            r=run(h,size)
            if r: res.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(res, open(OUT / 'turnover_capacity.json','w'), ensure_ascii=False, indent=1, default=float)
    print('DONE')
except Exception:
    print(traceback.format_exc())
