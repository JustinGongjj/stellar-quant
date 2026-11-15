"""验证缠论技术面特征的预测力（RankIC）。

关键约束：缠论工作站只提供**当前截面**的结构，没有历史快照。
因此无法做真正的时序 IC —— 这是个硬限制，必须如实说明。

能做的是：
  1. 当前截面特征 与 未来 5 日收益 的相关性（单期，样本量大）
  2. 分特征对比，看哪个结构指标最有区分度
"""
import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, json
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
OUT = DATA
C='CACHE/'; R=C+'wide2/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
op=M('open'); valid=np.load(R+'wide_valid.npy')
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
nd=len(op)
t=nd-8
while t>0 and valid[t].sum()<500: t-=1
print(f'截面 {t}, 有效 {int(valid[t].sum())}')

cf=pd.read_parquet(OUT / 'chanlun_feats.parquet')
H=5
fwd=np.asarray(op[t+1+H],dtype='float64')/np.asarray(op[t+1],dtype='float64')-1
dl=pd.Series(np.where(valid[t],fwd,np.nan), index=syms).dropna()
print(f'未来 {H} 日收益可得 {len(dl)} 只')

feats=['ch_trend','ch_pos','ch_pivots','ch_pos_pct','ch_last_signal','ch_seg_dir','ch_bi_dir','ch_sig_age']
res={}
for f in feats:
    if f not in cf.columns: continue
    s=cf.set_index('code')[f]
    j=pd.concat([s, dl], axis=1, keys=['f','y']).dropna()
    if len(j)<200: continue
    rx=j['f'].rank(); ry=j['y'].rank()
    ic=float(np.corrcoef(rx,ry)[0,1])
    # 分组收益
    q=pd.qcut(j['f'].rank(method='first'), 5, labels=False)
    g=j.groupby(q)['y'].mean()
    res[f]={'n':int(len(j)),'ic':round(ic,4),
            'q1':round(float(g.iloc[0])*100,2),'q5':round(float(g.iloc[-1])*100,2),
            'spread':round(float(g.iloc[-1]-g.iloc[0])*100,2)}
    print(f"{f:16s} n={len(j):5d} IC={ic:+.4f}  Q1={g.iloc[0]*100:+6.2f}% Q5={g.iloc[-1]*100:+6.2f}% 极差={res[f]['spread']:+5.2f}pp")

# 组合信号
cf2=cf.set_index('code')
comp=(cf2['ch_trend'].clip(-1,1)*0.4 + cf2['ch_pos'].clip(-1,1)*0.3
      + cf2['ch_last_signal']*0.2 + cf2['ch_seg_dir'].clip(-1,1)*0.1)
j=pd.concat([comp, dl], axis=1, keys=['f','y']).dropna()
if len(j)>200:
    ic=float(np.corrcoef(j['f'].rank(), j['y'].rank())[0,1])
    print(f"\n综合缠论分 IC={ic:+.4f}  n={len(j)}")
    res['composite']={'n':int(len(j)),'ic':round(ic,4)}

res['caveat']='缠论工作站只有当前截面快照，无法做时序 IC；以上为单期截面检验，不能外推为稳定 alpha'
json.dump(res, open(OUT / 'chanlun_ic.json','w'), ensure_ascii=False, indent=1)
print('\n', res['caveat'])
