import sys, time; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from stellarq.factors.build_wide import build
from stellarq.paths import CACHE, ROOT  # noqa: E402
t0=time.time()
r=build('CACHE/a_share_daily_v2.parquet', 'CACHE/wide2', cutoff='2026-09-30')
print(r, round(time.time()-t0), flush=True)
