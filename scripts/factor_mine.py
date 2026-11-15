"""因子挖掘主流程（基于干净宽表 wide2）。

流程：计算候选因子 → 截面检验（IC / ICIR / NW t / BH 校正 / 分层 / 换手）→ 输出报告。
所有因子只用「不复权价 + 成交额/换手」构造；跨除权的长周期动量跳过（需要复权，暂缺）。
"""
import sys, time, gc, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
import stellarq.factors.alpha_fast as af
from stellarq.validation.fast import cs_prep
from stellarq.validation.factor_test import bh_adjust, pvalue_from_t, newey_west_t
from stellarq.paths import DATA, ROOT, WIDE  # noqa: E402
OUT = DATA
def w(m):
    with open(OUT / 'factor_mine.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'factor_mine.log','w').close()
t0=time.time()
R='WIDE/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
o,h,l,c,v,amt,turn,cap,vwap = M('open'),M('high'),M('low'),M('close'),M('volume'),M('amount'),M('turnover'),M('float_mktcap'),M('avg_price')
valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy', allow_pickle=True); syms=np.load(R+'wide_syms.npy', allow_pickle=True)
nd,ns=c.shape
w(f'panel {nd} x {ns}')

def col(f, i0, i1): return np.asarray(f[i0:i1], dtype=np.float64)

# ---------- 标签：T+1 开盘买入、T+1+h 开盘卖出
def label(hh):
    lab=np.full((nd,ns), np.nan, dtype=np.float32)
    for i0 in range(0, nd-1-hh, 800):
        i1=min(i0+800, nd-1-hh)
        buy=col(o,i0+1,i1+1); sell=col(o,i0+1+hh,i1+1+hh)
        lab[i0:i1]=sell/buy-1.0
    return np.where(valid, lab, np.nan)

# ---------- 因子定义（在 [i0-w, i1) 窗口内按块计算）
def make(name):
    import re
    m=re.match(r'([A-Z_]+?)(\d*)$', name)
    base, wn = (m.group(1), m.group(2)) if m else (name, '')
    ww=int(wn) if wn else 0
    def f(i0,i1):
        s=max(0,i0-ww)
        cc=col(c,s,i1); k=i1-s; off=i0-s
        if base=='KMID': return (col(c,i0,i1)-col(o,i0,i1))/col(o,i0,i1)
        if base=='KLEN': return (col(h,i0,i1)-col(l,i0,i1))/col(o,i0,i1)
        if base=='CLOSE_POS': return (col(c,i0,i1)-col(l,i0,i1))/(col(h,i0,i1)-col(l,i0,i1)+1e-12)
        if base=='VWAP_GAP': return col(c,i0,i1)/col(vwap,i0,i1)-1.0
        if base=='LN_CAP': return np.log1p(col(cap,i0,i1))
        if base=='Overnight':
            cc=col(c,i0,i1); oo=col(o,i0,i1); r=np.full_like(cc,np.nan); r[1:]=np.log(oo[1:]/cc[:-1]); return r
        if base=='Intraday': return np.log(col(c,i0,i1)/col(o,i0,i1))
        if base=='RET': return col(c,i0-ww,i1-ww)/col(c,i0,i1)-1.0 if i0>=ww else np.full((i1-i0,ns),np.nan)
        if base=='MA':
            ext=cc; mm=af.roll_mean(ext,ww,max(2,ww//2)); return mm[off:]/col(c,i0,i1)-1.0
        if base=='STD':
            ext=cc; r=np.full_like(ext,np.nan); r[1:]=ext[1:]/ext[:-1]-1.0
            return af.roll_std(r,ww,max(2,ww//2))[off:]
        if base=='MIN':
            ext=col(l,s,i1); return af.roll_min_ignore(ext,ww,max(2,ww//2))[off:]/col(c,i0,i1)-1.0
        if base=='MAX':
            ext=col(h,s,i1); return af.roll_max_ignore(ext,ww,max(2,ww//2))[off:]/col(c,i0,i1)-1.0
        if base=='TURN':
            ext=col(turn,s,i1); return af.roll_mean(ext,ww,max(2,ww//2))[off:]/100.0
        if base=='VMA':
            ext=col(v,s,i1); cur=col(v,i0,i1); return af.roll_mean(ext,ww,max(2,ww//2))[off:]/(cur+1e-12)-1.0
        if base=='AMT_MA':
            ext=np.log1p(col(amt,s,i1)); return af.roll_mean(ext,ww,max(2,ww//2))[off:]
        if base=='AMT_RATIO':
            ext=np.log1p(col(amt,s,i1)); return ext[off:]-af.roll_mean(ext,ww,max(2,ww//2))[off:]
        if base=='AMIHUD':
            cc2=col(c,s,i1); r=np.full_like(cc2,np.nan); r[1:]=cc2[1:]/cc2[:-1]-1.0
            return af.roll_mean(np.abs(r)/(np.log1p(col(amt,s,i1))+1e-12),ww,max(2,ww//2))[off:]
        if base=='VSTD':
            ext=np.log1p(col(v,s,i1)); return af.roll_std(ext,ww,max(2,ww//2))[off:]
        if base=='CORR':
            cc2=col(c,s,i1); r=np.full_like(cc2,np.nan); r[1:]=cc2[1:]/cc2[:-1]-1.0
            vv=col(v,s,i1); vr=np.full_like(vv,np.nan); vr[1:]=vv[1:]/vv[:-1]-1.0
            return af.roll_corr(r, np.log1p(np.abs(vr)), ww, max(3,ww//3))[off:]
        if base=='CNTD':
            cc2=col(c,s,i1); r=np.full_like(cc2,np.nan); r[1:]=cc2[1:]/cc2[:-1]-1.0
            up=np.where(np.isfinite(r),(r>0).astype(float),np.nan)
            p=af.roll_mean(up,ww,max(2,ww//2))[off:]
            return 2.0*p-1.0
        if base=='VSUMD':
            vv=col(v,s,i1); dv=np.full_like(vv,np.nan); dv[1:]=np.diff(vv,axis=0)
            pos=af.roll_mean(np.where(np.isfinite(dv),np.maximum(dv,0),np.nan),ww,max(2,ww//2))[off:]
            neg=af.roll_mean(np.where(np.isfinite(dv),np.maximum(-dv,0),np.nan),ww,max(2,ww//2))[off:]
            return (pos-neg)/(pos+neg+1e-12)
        if base=='HL_RANGE':
            cc2=col(c,s,i1); pc=np.full_like(cc2,np.nan); pc[1:]=cc2[:-1]
            return (col(h,s,i1)-col(l,s,i1))/pc
        raise KeyError(name)
    return f

STATIC=['KMID','KLEN','CLOSE_POS','VWAP_GAP','LN_CAP','Overnight','Intraday']
ROLL=[f'{b}{w}' for w in (5,10,20,60) for b in ('MA','STD','MIN','MAX','RET','TURN','VMA','AMT_MA','AMT_RATIO','VSTD','CORR','CNTD','VSUMD')]
WINDOWED=['AMIHUD']
NAMES=STATIC+ROLL+WINDOWED
w(f'candidate factors {len(NAMES)}')

lab=label(5)
w(f'label ready {round(time.time()-t0,1)}s')

WARM=65; BLOCK=200; MIN_N=300
def screen(name):
    fn=make(name)
    ics=[]; groups=[]; prev=None; turns=[]
    for i1 in list(range(WARM, nd, BLOCK))+[nd]:
        i0=max(WARM, i1-BLOCK)
        if i1<=i0: continue
        f=np.where(valid[i0:i1], fn(i0,i1), np.nan)
        y=np.where(valid[i0:i1], lab[i0:i1], np.nan)
        ff=cs_prep(f)
        both=np.isfinite(ff)&np.isfinite(y)
        ranks=np.full(f.shape,np.nan)
        for r in range(f.shape[0]):
            m=both[r]; n=int(m.sum())
            if n<MIN_N: continue
            idx=np.where(m)[0]; order=idx[np.argsort(ff[r,idx],kind='mergesort')]
            ranks[r,order]=np.arange(1,n+1)
            yy=y[r,order]; cuts=np.linspace(0,n,6).astype(int)
            groups.append([float(np.nanmean(yy[cuts[g]:cuts[g+1]])) for g in range(5)])
            a=ff[r,m]; b=y[r,m]
            ra=np.argsort(np.argsort(a,kind='mergesort'),kind='mergesort').astype(float)
            rb=np.argsort(np.argsort(b,kind='mergesort'),kind='mergesort').astype(float)
            sa,sb=ra.std(),rb.std()
            if sa>0 and sb>0: ics.append(float(((ra-ra.mean())*(rb-rb.mean())).mean()/(sa*sb)))
        if prev is not None:
            vals=[]
            for r in range(ranks.shape[0]):
                m=np.isfinite(prev[r])&np.isfinite(ranks[r])
                if m.sum()<MIN_N: continue
                a,b=prev[r][m],ranks[r][m]
                if a.std()==0 or b.std()==0: continue
                vals.append(1-float(np.corrcoef(a,b)[0,1]))
            if vals: turns.append(float(np.nanmean(vals)))
        prev=ranks
    return np.array(ics), np.array(groups), (float(np.nanmean(turns)) if turns else np.nan)

rows=[]
for name in NAMES:
    t1=time.time()
    try:
        ic,g,tv=screen(name)
    except KeyError:
        continue
    if len(ic)<100: 
        w(f'skip {name}'); continue
    sd=float(ic.std(ddof=1)); mu=float(ic.mean())
    ls=g[:,-1]-g[:,0]; ls=ls[np.isfinite(ls)]
    gm=np.nanmean(g,axis=0)
    mono=float(np.corrcoef(np.arange(1,6),gm)[0,1]) if np.isfinite(gm).all() and np.std(gm)>0 else np.nan
    rows.append({'factor':name,'n':len(ic),'ic_mean':mu,'ic_std':sd,'icir':mu/sd if sd>0 else np.nan,
                 't_nw':newey_west_t(pd.Series(ic)),'positive_ratio':float((ic>0).mean()),
                 'ls_mean':float(ls.mean()) if len(ls) else np.nan,
                 'ls_t':float(ls.mean()/(ls.std(ddof=1)/np.sqrt(len(ls)))) if len(ls)>2 and ls.std(ddof=1)>0 else np.nan,
                 'monotonicity':mono,'turnover':tv})
    w(f'{name:12s} IC {mu:+.4f} ICIR {rows[-1]["icir"]:+.3f} t {rows[-1]["t_nw"]:+.1f} {round(time.time()-t1,1)}s')
    del ic,g
    gc.collect()
rep=pd.DataFrame(rows)
rep['p_value']=[pvalue_from_t(t,max(int(n)-1,1)) for t,n in zip(rep['t_nw'],rep['n'])]
rep['q_bh']=bh_adjust(rep['p_value'].fillna(1).tolist()); rep['abs_icir']=rep['icir'].abs()
rep=rep.sort_values('abs_icir',ascending=False)
rep.to_csv(OUT / 'factor_mine_h5.csv', index=False)
w(rep.head(20)[['factor','n','ic_mean','icir','t_nw','q_bh','ls_t','monotonicity','turnover']].round(4).to_string())
w('DONE '+str(round(time.time()-t0)))
