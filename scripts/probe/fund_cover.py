import sys, glob; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import pandas as pd, numpy as np
from stellarq.factors.event_fund import fundamental_panel
files=sorted(glob.glob(str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/fund/*.parquet'))
fp=fundamental_panel(files, pd.Timestamp('2026-09-30'))
print('rows', len(fp))
print('symbol 前缀分布:')
print(fp['symbol'].str[:2].value_counts().head(10))
print('notice 年份:', fp['notice'].dt.year.value_counts().sort_index().to_dict())
print('report 季度数', fp['report'].dt.to_period('Q').nunique())
print('BPS 有效比例', float(fp['BPS'].notna().mean()), 'ROE 有效', float(fp['WEIGHTAVG_ROE'].notna().mean()))
