"""验证五碗面打分的预测力：用历史截面算 RankIC。"""
import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, json
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
OUT = DATA
C='CACHE/'; R=C+'wide2/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
close=M('close'); turn=M('turnover'); amt=M('amount'); cap=M('float_mktcap'); op=M('open')
valid=np.load(R+'wide_valid.npy')
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
nd=len(close)
H=5
ics={k:[] for k in ['tech','sent','fund','fundm','score']}
for t in range(300, nd-H-1, 10):
    c=np.asarray(close[t-60:t+1],dtype='float64'); a=np.asarray(amt[t-20:t+1],dtype='float64')
    tu=np.asarray(turn[t-60:t+1],dtype='float64'); cp=np.asarray(cap[t],dtype='float64')
    cur=c[-1]; m=valid[t]&np.isfinite(cur)&np.isfinite(cp)
    if m.sum()<500: continue
    def pct(x):
        out=np.full(len(x),np.nan); ok=np.isfinite(x)&m
        if ok.sum()<50: return out
        out[ok]=pd.Series(x[ok]).rank(pct=True).to_numpy()*100
        return out
    ma20=np.nanmean(c[-20:],axis=0)
    tech=0.5*pct((cur-ma20)/ma20)+0.5*pct(cur/c[-21]-1)
    ret=np.full_like(c,np.nan); ret[1:]=c[1:]/c[:-1]-1
    sent=0.5*pct(-np.nanmean(tu[-5:],axis=0))+0.5*pct(-np.nanstd(ret[-20:],axis=0))
    fund=pct(-np.log(np.nanmean(a[-5:],axis=0)/np.nanmean(a[-20:],axis=0)+1e-12))
    fundm=pct(1.0/np.maximum(cp,1e-9))
    score=np.nanmean(np.vstack([tech,sent,fund,fundm]),axis=0)
    fwd=np.asarray(op[t+1+H],dtype='float64')/np.asarray(op[t+1],dtype='float64')-1
    fwd=np.where(valid[t],fwd,np.nan)
    ok=m&np.isfinite(fwd)
    if ok.sum()<300: continue
    for k,v in [('tech',tech),('sent',sent),('fund',fund),('fundm',fundm),('score',score)]:
        x=v[ok]; y=fwd[ok]
        if not (np.isfinite(x).all()): continue
        rx=pd.Series(x).rank(); ry=pd.Series(y).rank()
        ics[k].append(float(np.corrcoef(rx,ry)[0,1]))
res={}
for k,v in ics.items():
    if len(v)<10: continue
    arr=np.array(v)
    res[k]={'n':len(arr),'ic':round(float(arr.mean()),4),'icir':round(float(arr.mean()/arr.std(ddof=1)),3),
            'win':round(float((arr>0).mean()),3)}
    print(f'{k:8s} IC={res[k]["ic"]:+.4f} ICIR={res[k]["icir"]:+.3f} 胜率={res[k]["win"]*100:.1f}% ({res[k]["n"]} 期)')
json.dump(res, open(OUT / 'five_faces_test.json','w'), ensure_ascii=False, indent=1)
