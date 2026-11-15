"""量化幸存者偏差的方向与量级。

思路：退市股在「退市前」往往经历大幅下跌。它们不在面板里意味着：
  1. 做空/回避类信号被系统性低估；
  2. 选股组合不会被这些股票拖累，回测收益被高估。

可测的部分：用交易所给的退市日期，统计「退市前 1 年/半年」沪深整体跌幅，
把缺失股的期望收益近似为「退市前表现」。由于拿不到它们的历史行情（东财封了、腾讯限流），
这里改用**保守假设区间**做敏感性：假设缺失股在其在场期的年化收益为 −30% / −50% / −70%。
"""
import json
import pandas as pd, numpy as np
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
d=pd.read_csv(OUT / 'delisted_check.csv')
d['delist_date']=pd.to_datetime(d['delist_date'], format='%Y%m%d', errors='coerce')
d['list_date']=pd.to_datetime(d['list_date'], format='%Y%m%d', errors='coerce')
recent=d[d['delist_date']>=pd.Timestamp('2012-01-01')]
print('2012 年后退市（沪市）', len(recent))
# 我们面板的期数（样本外 105 期，每期 50 只），粗估「每期被漏掉的股票占比」
total_syms=5027
missing_share=len(recent)/total_syms
print(f'缺股比例 {missing_share:.2%}')
rows=[]
for ann in (-0.30,-0.50,-0.70):
    # 假设这些股票在市场中的权重与其数量占比一致，则组合的期望收益被高估约 missing_share × |ann|
    rows.append({'缺失股年化': ann, '组合年化高估(pp)': round(-ann*missing_share*100,2)})
df=pd.DataFrame(rows)
print(df.to_string(index=False))
json.dump({'sse_delisted_total':int(len(d)),'delisted_since_2012':int(len(recent)),
           'panel_covers_delisted':0,'missing_share':float(missing_share),
           'sensitivity':df.to_dict('records')}, open(OUT / 'survivorship.json','w'), ensure_ascii=False, indent=1)
