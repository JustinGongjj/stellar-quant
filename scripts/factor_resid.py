"""因子残差化 + 换手抑制（回答两个问题）。
from stellarq.paths import DATA, ROOT, WIDE  # noqa: E402

Q1 VSTD20 这类量能因子，是不是「另一个规模因子/波动率因子」？
   → 截面回归 VSTD20 ~ ln(流通市值) + 60日波动 + 换手率，取残差再看 IC。
Q2 换手 0.93 太高，排序平滑能不能把费用压下来而 IC 不变？
   → 比较：原始打分 / 3 期移动平均打分 / top-k 缓冲区（进前 60 才买，跌出 80 才卖）。
"""
import sys, time, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'resid.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'resid.log','w').close()
try:
    import numpy as np, pandas as pd
    import stellarq.factors.alpha_fast as af
    from stellarq.validation.fast import cs_prep, rank_ic, ic_stats_fast
    t0=time.time()
    R='WIDE/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o,c,v,amt,turn,cap = M('open'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap')
    valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    nd,ns=c.shape; H=6
    lab=np.full((nd,ns),np.nan,dtype=np.float32)
    for i0 in range(0,nd-1-H,600):
        i1=min(i0+600,nd-1-H)
        lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
    lab=np.where(valid,lab,np.nan)
    # 只算 3 个代表因子，控制时间
    TARGETS={'VSTD20':20,'AMT_RATIO60':60,'MA60':60}
    res={}
    for name,ww in TARGETS.items():
        F=np.full((nd,ns),np.nan,dtype=np.float32)
        for i0 in range(70,nd,300):
            i1=min(i0+300,nd); s=max(0,i0-ww); off=i0-s
            if name=='VSTD20':
                ext=np.log1p(np.asarray(v[s:i1],dtype=np.float64))
                F[i0:i1]=af.roll_std(ext,20,max(2,10))[off:]
            elif name=='AMT_RATIO60':
                ext=np.log1p(np.asarray(amt[s:i1],dtype=np.float64))
                F[i0:i1]=ext[off:]-af.roll_mean(ext,60,30)[off:]
            else:
                ext=np.asarray(c[s:i1],dtype=np.float64)
                F[i0:i1]=af.roll_mean(ext,60,30)[off:]/np.asarray(c[i0:i1],dtype=np.float64)-1.0
        F=np.where(valid,F,np.nan)
        # 暴露：ln 市值、60日波动、换手
        VOL60=np.full((nd,ns),np.nan,dtype=np.float32)
        for i0 in range(70,nd,300):
            i1=min(i0+300,nd); s=max(0,i0-60)
            cc=np.asarray(c[s:i1],dtype=np.float64); r=np.full_like(cc,np.nan); r[1:]=cc[1:]/cc[:-1]-1.0
            VOL60[i0:i1]=af.roll_std(r,60,30)[i0-s:]
        VOL60=np.where(valid,VOL60,np.nan)
        LNCAP=np.where(valid, np.log1p(np.asarray(cap,dtype=np.float64)), np.nan)
        TURN=np.where(valid, np.asarray(turn,dtype=np.float64)/100.0, np.nan)
        # 逐截面 OLS 取残差
        resid=np.full((nd,ns),np.nan,dtype=np.float32)
        for t in range(70,nd):
            m=np.isfinite(F[t])&np.isfinite(lab[t])&np.isfinite(LNCAP[t])&np.isfinite(VOL60[t])&np.isfinite(TURN[t])
            if m.sum()<300: continue
            X=np.column_stack([np.ones(m.sum()), LNCAP[t][m], VOL60[t][m], TURN[t][m]])
            b,*_=np.linalg.lstsq(X, F[t][m], rcond=None)
            resid[t][m]=F[t][m]-X@b
        raw_ic=rank_ic(cs_prep(np.nan_to_num(F,nan=np.nan)), lab, 300)
        res_ic=rank_ic(cs_prep(resid), lab, 300)
        a=ic_stats_fast(raw_ic); b2=ic_stats_fast(res_ic)
        res[name]={'raw':a,'resid':b2,'retained_icir':(b2['icir']/a['icir']) if a.get('icir') else None}
        w(f"{name}: 原始 ICIR {a['icir']:+.3f} (IC {a['ic_mean']:+.4f}) | 残差化后 ICIR {b2['icir']:+.3f} (IC {b2['ic_mean']:+.4f}) | 保留 {res[name]['retained_icir']:.0%}")
    json.dump(res, open(OUT / 'resid.json','w'), ensure_ascii=False, indent=1, default=float)
    w('DONE '+str(round(time.time()-t0)))
except Exception:
    w(traceback.format_exc())
