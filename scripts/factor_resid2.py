"""因子残差化（内存安全版）：判断量能类因子是不是「规模/波动」的马甲。
from stellarq.paths import DATA, ROOT, WIDE  # noqa: E402

做法：逐日截面把因子对 [ln 流通市值, 60 日波动, 换手率] 做 OLS，取残差，
再看残差因子的 IC / ICIR 保留多少。分块处理，避免整块 float64。
"""
import sys, time, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'resid.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'resid.log','w').close()
try:
    import numpy as np
    t0=time.time()
    R='WIDE/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o,c,v,turn,cap = M('open'),M('close'),M('volume'),M('turnover'),M('float_mktcap')
    valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    nd,ns=c.shape; H=6
    lab=np.full((nd,ns),np.nan,dtype=np.float32)
    for i0 in range(0,nd-1-H,600):
        i1=min(i0+600,nd-1-H)
        lab[i0:i1]=np.asarray(o[i0+1+H:i1+1+H],dtype=np.float64)/np.asarray(o[i0+1:i1+1],dtype=np.float64)-1.0
    lab=np.where(valid,lab,np.nan)

    def build(name, t0_, t1_):
        """在 [t0_, t1_) 计算因子值，窗口向前多取 60 日。"""
        ww=20 if name=='VSTD20' else 60
        s=max(0,t0_-ww)
        out=np.full((t1_-t0_, ns), np.nan)
        for a in range(t0_, t1_, 100):
            b=min(a+100, t1_)
            sa=max(0,a-ww); off=a-sa
            if name=='VSTD20':
                ext=np.log1p(np.asarray(v[sa:b],dtype=np.float64))
                out[a-t0_:b-t0_]=ext[off:]-np.cumsum(ext,axis=0)[off:]/np.arange(off+1,off+1+(b-sa-off))[:,None] if False else _std(ext, 20)[off:]
            elif name=='AMT_RATIO60':
                ext=np.log1p(np.asarray(v[sa:b],dtype=np.float64)*np.asarray(c[sa:b],dtype=np.float64))
                out[a-t0_:b-t0_]=ext[off:]-_mean(ext,60)[off:]
            else:  # MA60
                ext=np.asarray(c[sa:b],dtype=np.float64)
                out[a-t0_:b-t0_]=_mean(ext,60)[off:]/np.asarray(c[a:b],dtype=np.float64)-1.0
        return out

    def _mean(x, k):
        out=np.full_like(x,np.nan); cs=np.cumsum(np.nan_to_num(x),axis=0); cnt=np.cumsum(np.isfinite(x),axis=0)
        for i in range(k-1,len(x)):
            lo=i-k
            s=cs[i]-(cs[lo] if lo>=0 else 0); n=cnt[i]-(cnt[lo] if lo>=0 else 0)
            out[i]=s/n
        return out

    def _std(x, k):
        m=_mean(x,k); out=np.full_like(x,np.nan)
        for i in range(k-1,len(x)):
            seg=x[i-k+1:i+1]
            out[i]=np.nanstd(seg,axis=0,ddof=1)
        return out

    res={}
    for name in ('VSTD20','AMT_RATIO60','MA60'):
        ics_raw=[]; ics_res=[]; ret_raw=[]; ret_res=[]
        for a in range(70, nd-H-1, 100):
            b=min(a+100, nd-H-1)
            F=build(name, a, b)
            VOL=np.full_like(F, np.nan)
            for k in range(b-a):
                t=a+k
                lo=max(0,t-60)
                cc=np.asarray(c[lo:t+1],dtype=np.float64)
                r=np.full_like(cc,np.nan); r[1:]=cc[1:]/cc[:-1]-1.0
                VOL[k]=np.nanstd(r,axis=0,ddof=1)
            LNCAP=np.log1p(np.asarray(cap[a:b],dtype=np.float64))
            TN=np.asarray(turn[a:b],dtype=np.float64)/100.0
            for k in range(b-a):
                t=a+k
                m=np.isfinite(F[k])&np.isfinite(lab[t])&np.isfinite(LNCAP[k])&np.isfinite(VOL[k])&np.isfinite(TN[k])
                if m.sum()<300: continue
                X=np.column_stack([np.ones(m.sum()),LNCAP[k][m],VOL[k][m],TN[k][m]])
                beta,*_=np.linalg.lstsq(X,F[k][m],rcond=None)
                e=F[k][m]-X@beta
                yr=lab[t][m]
                def rk(z):
                    rr=np.empty(len(z)); rr[np.argsort(z,kind='mergesort')]=np.arange(len(z)); return rr
                ics_raw.append(float(np.corrcoef(rk(F[k][m]),rk(yr))[0,1]))
                ics_res.append(float(np.corrcoef(rk(e),rk(yr))[0,1]))
            w(f'{name} t={b} raw_ic {np.mean(ics_raw):+.4f} resid_ic {np.mean(ics_res):+.4f} {round(time.time()-t0)}s')
        a_raw=np.array(ics_raw); a_res=np.array(ics_res)
        res[name]={'n':int(len(a_raw)),'raw_ic':float(a_raw.mean()),'raw_icir':float(a_raw.mean()/a_raw.std(ddof=1)),
                   'resid_ic':float(a_res.mean()),'resid_icir':float(a_res.mean()/a_res.std(ddof=1)),
                   'icir_retained':float((a_res.mean()/a_res.std(ddof=1))/(a_raw.mean()/a_raw.std(ddof=1)))}
        w(json.dumps(res[name], ensure_ascii=False))
    json.dump(res, open(OUT / 'resid.json','w'), ensure_ascii=False, indent=1)
    w('DONE')
except Exception:
    w(traceback.format_exc())
