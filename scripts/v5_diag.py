"""诊断：基本面因子在训练期与测试期的表现差异（解释 v5 为何输给 v4）。"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
C='CACHE/'
X=np.load(C+'v4_X.npy'); E=np.load(C+'v5_extra.npy'); y=np.load(C+'v4_y.npy'); d=np.load(C+'v4_d.npy')
F8=['f_bp','f_ep','f_growth','f_rev_grow','f_cfo','f_rev_acc','f_margin','f_roe']
V4=['KMID','LN_CAP','MA10','MA20','MA60','MA20_resid','MA60_resid','STD20','MIN5','MIN20','RET20','TURN5',
    'VSTD20','VSTD20_resid','AMT_RATIO60','AMT_RATIO60_resid','VMA60','VSUMD60','CORR20']
uniq=np.unique(d); cut=uniq[int(len(uniq)*0.85)]
te=d>=cut
out=[]
for j,name in enumerate(F8):
    col=E[:,j]
    def ic(mask):
        vals=[]
        for u in np.unique(d[mask]):
            m=mask&(d==u)
            c=col[m]; yy=y[m]
            ok=np.isfinite(c)&np.isfinite(yy)
            if ok.sum()<200: continue
            ra=np.argsort(np.argsort(c[ok],kind='mergesort')).astype(float)
            rb=np.argsort(np.argsort(yy[ok],kind='mergesort')).astype(float)
            if ra.std()>0 and rb.std()>0:
                vals.append(float(((ra-ra.mean())*(rb-rb.mean())).mean()/(ra.std()*rb.std())))
        return (np.mean(vals) if vals else np.nan), len(vals)
    a,na=ic(~te); b,nb=ic(te)
    out.append({'factor':name,'ic_train':a,'ic_test':b,'n_train':na,'n_test':nb})
for j,name in enumerate(V4):
    col=X[:,j]
    def ic2(mask):
        vals=[]
        for u in np.unique(d[mask]):
            m=mask&(d==u); c=col[m]; yy=y[m]
            ok=np.isfinite(c)&np.isfinite(yy)
            if ok.sum()<200: continue
            ra=np.argsort(np.argsort(c[ok],kind='mergesort')).astype(float)
            rb=np.argsort(np.argsort(yy[ok],kind='mergesort')).astype(float)
            if ra.std()>0 and rb.std()>0:
                vals.append(float(((ra-ra.mean())*(rb-rb.mean())).mean()/(ra.std()*rb.std())))
        return (np.mean(vals) if vals else np.nan)
    out.append({'factor':name,'ic_train':ic2(~te),'ic_test':ic2(te),'n_train':np.nan,'n_test':np.nan})
df=pd.DataFrame(out)
df['decay']=df['ic_test']/df['ic_train']
print(df.round(4).to_string(index=False))
df.to_csv('DATA/v5_factor_decay.csv', index=False)
