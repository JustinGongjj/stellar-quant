import numpy as np, pandas as pd, pyarrow.parquet as pq
F=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/a_share_daily_v2.parquet'
pf=pq.ParquetFile(F)
tot=0; bad=0; syms_with_bad=0; n_sym=0
for i in range(pf.num_row_groups):
    d=pf.read_row_group(i, columns=['symbol','date','close']).to_pandas()
    d=d.sort_values(['symbol','date'])
    r=d.groupby('symbol')['close'].pct_change().abs()*100
    lim=np.where(d['symbol'].str.startswith(('688','689','300','301')),20.0,10.0)
    over=(r.to_numpy()>np.asarray(lim)+0.6)
    over&=np.isfinite(r.to_numpy())
    tot+=len(d); bad+=int(over.sum())
    g=pd.DataFrame({'s':d['symbol'],'o':over}).groupby('s')['o'].any()
    syms_with_bad+=int(g.sum()); n_sym+=len(g)
    print('rg',i,'rows',len(d),'bad',int(over.sum()), flush=True)
print('总计', tot, '超限', bad, round(bad/tot*100,4),'% | 含超限股票', syms_with_bad, '/', n_sym)
