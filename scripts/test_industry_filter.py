"""检验：产业风险过滤能否提升量化模型的表现。

方法：
  在量化模型（v4）的 top60 持仓里，剔除「产业风险 > 阈值」的股票，
  看超额与 IR 是否改善。

局限要先说：产业风险只覆盖 21 只股票，而 top60 里可能一只都碰不到，
那样这个检验就是空的。
"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
OUT = DATA
C='CACHE/'; R=C+'wide2/'
RISK=json.load(open(OUT / 'industry_risk.json',encoding='utf-8'))['risk']
st=np.load(C+'v4_st.npy'); dte=np.load(C+'v4_dte.npy'); ste=np.load(C+'v4_ste.npy')
o=np.load(R+'wide_open.npy',mmap_mode='r'); valid=np.load(R+'wide_valid.npy')
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
H=6
hits=0; total=0
rows=[]
for u in list(np.unique(dte)):
    uu=int(u)
    sc=pd.Series(st[dte==u], index=syms[ste[dte==u]]).dropna()
    cand=list(sc.sort_values(ascending=False).head(60).index)
    total+=len(cand)
    hits+=sum(1 for c in cand if c in RISK)
    rp=np.asarray(o[uu+1+H],dtype=np.float64)/np.asarray(o[uu+1],dtype=np.float64)-1
    rp=pd.Series(np.where(valid[uu],rp,np.nan), index=syms)
    r=rp.reindex(cand).dropna()
    filt=[c for c in r.index if RISK.get(c,0)<=0.3]
    rf=rp.reindex(filt).dropna()
    if len(r)>=20:
        rows.append({'date':uu,'base':float(r.mean()),'filt':float(rf.mean()) if len(rf)>=10 else np.nan,
                     'bench':float(rp.dropna().mean()),'n_removed':len(r)-len(rf)})
print(f'top60 中命中产业风险的次数: {hits} / {total} ({hits/max(total,1)*100:.1f}%)')
df=pd.DataFrame(rows).dropna()
print(f'有效期数: {len(df)}  平均剔除 {df["n_removed"].mean():.1f} 只')
def perf(x,label):
    x=np.asarray(x); ann=x.mean()*252/H; vol=x.std(ddof=1)*np.sqrt(252/H)
    nav=np.cumprod(1+x); dd=float((nav/np.maximum.accumulate(nav)-1).min())
    return {'label':label,'ann':float(ann),'vol':float(vol),'ir':float(ann/vol) if vol>0 else None,'dd':dd}
if len(df)>10:
    for lbl,col in [('原模型','base'),('加产业过滤','filt')]:
        ex=df[col]-df['bench']
        p=perf(ex,lbl)
        print(f"  {p['label']:10s} 超额年化 {p['ann']*100:+6.2f}% 波动 {p['vol']*100:5.2f}% IR {p['ir']:+.2f}")
    json.dump({'coverage':{'hits':hits,'total':total,'stocks_with_risk':len(RISK)},
               'n_periods':int(len(df)),'avg_removed':float(df['n_removed'].mean())},
              open(OUT / 'industry_filter_test.json','w'), ensure_ascii=False, indent=1)
else:
    print('样本不足，无法检验')
    json.dump({'coverage':{'hits':hits,'total':total,'stocks_with_risk':len(RISK)},
               'n_periods':int(len(df)),'note':'top60 中几乎不含产业风险股票，无法检验'},
              open(OUT / 'industry_filter_test.json','w'), ensure_ascii=False, indent=1)
