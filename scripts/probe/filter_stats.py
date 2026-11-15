import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
from stellarq.data.sources import price_limit_mask
import pyarrow.parquet as pq, pandas as pd
pf=pq.ParquetFile(str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/a_share_daily_v2.parquet')
keep=0; total=0
for i in range(pf.num_row_groups):
    d=pf.read_row_group(i, columns=['symbol','date','close']).to_pandas()
    m=price_limit_mask(d)
    keep+=int(m.sum()); total+=len(d)
    print('rg',i, round(float(m.mean())*100,3),'%', flush=True)
print('保留', keep, '/', total, round(keep/total*100,3),'%')
