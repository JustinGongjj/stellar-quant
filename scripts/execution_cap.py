"""执行限速：当日成交额占比上限（ADV cap）对收益与成本的影响。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

逻辑：单票下单金额不超过其成交额的 x%，放不下的部分延到下一交易日。
  * 好处：冲击成本可控；
  * 代价：仓位到位变慢（这里按「未成交部分按次日开盘价补」，近似为收益延迟 + 部分暴露缺失）。
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
    dates=np.load(R+'wide_dates.npy',allow_pickle=True); H=6
    COMM=0.00025; TRANS=0.00001; K=0.0137   # 见 scripts/probe/impact_calib.py：K=0.3 会高估 20 倍
    def run(size, cap_frac, top_n=60, hold=50):
        prev=set(); per=[]
        for u in list(np.unique(dte)):
            uu=int(u)
            sc=pd.Series(st[dte==u], index=syms_str[ste[dte==u]]).dropna()
            if len(sc)<100: continue
            cand=list(sc.sort_values(ascending=False).head(top_n).index)
            held=list(dict.fromkeys(cand))[:hold]
            # 按 ADV 上限计算「当期能建多少仓」
            adv=pd.Series(np.asarray(amt[uu],dtype=np.float64), index=syms_str).reindex(held).fillna(1e4)
            per_stock=size/len(held)
            if cap_frac:
                fillable=np.minimum(per_stock, adv.values*cap_frac)
                fill_ratio=float(np.mean(fillable/per_stock))         # 当期平均建仓比例
            else:
                fill_ratio=1.0
            rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1.0
            rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms_str)
            r=rp.reindex(held).dropna()
            if len(r)<20: continue
            gross=float(r.mean())*fill_ratio           # 未成交部分不计收益（保守）
            new=set(r.index); turn_=len(new^prev)/max(len(new)+len(prev),1)*fill_ratio; prev=new
            q=np.minimum(per_stock, adv.values*cap_frac) if cap_frac else per_stock
            # 量纲：q 与 adv 都是「元」；此前把 adv 又除 1e4，导致冲击成本被放大 1e4 倍
            impact=float(np.nanmean(K*np.sqrt(np.maximum(q,0)/np.maximum(adv.values,1.0))))
            cost=turn_*(COMM+TRANS+impact)
            per.append({'gross':gross,'net':gross-cost,'bench':float(rp.dropna().mean()),'fill':fill_ratio,'impact':impact,'turn':turn_})
        pf=pd.DataFrame(per)
        ov=pf['net'].rolling(5).mean().dropna(); ex=(pf['net']-pf['bench']).rolling(5).mean().dropna()
        return {'size_yi':round(size/1e8,2),'cap':cap_frac,'fill':float(pf['fill'].mean()),
                'ann_net':float(ov.mean()*10.08),'ann_excess':float(ex.mean()*10.08),
                'ann_excess_surv_adj':float(ex.mean()*10.08-0.0122),
                'ir':float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08)),
                'impact_bp':float(pf['impact'].mean()*1e4),'turnover':float(pf['turn'].mean())}
    res=[]
    for size in (1e8, 5e8, 2e9):
        for cap in (None, 0.05, 0.02, 0.01):
            res.append(run(size,cap))
            print(json.dumps(res[-1], ensure_ascii=False), flush=True)
    json.dump(res, open(OUT / 'execution_cap.json','w'), ensure_ascii=False, indent=1, default=float)
    print('DONE')
except Exception:
    print(traceback.format_exc())
