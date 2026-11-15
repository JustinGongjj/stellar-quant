"""构建美林时钟历史相位，并检验它在各阶段的可交易资产表现。"""
from stellarq.paths import DATA, ROOT  # noqa: E402
import sys, json, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    from stellarq.macro.clock import macro_panel, classify, PHASES
    from stellarq.data.tencent import Tencent
    panel=macro_panel('2012-01-01')
    w(f'macro panel {panel.shape}, {panel.index.min().date()} -> {panel.index.max().date()}')
    w('columns '+', '.join(panel.columns))
    clock=classify(panel, lookback=3)
    clock.to_csv(OUT / 'clock_history.csv')
    w('phase counts: '+json.dumps(clock["phase"].value_counts().to_dict(), ensure_ascii=False))
    w('latest phase: '+str(clock['phase'].iloc[-1])+' | '+str(clock.index[-1].date()))
    # 资产表现：沪深300(399300)、中证500(000905)、国债指数代理用 000012? 用大盘/中盘/黄金
    t=Tencent()
    assets={}
    for name,code in [('沪深300','399300'),('中证500','000905'),('中证1000','000852')]:
        d=t.daily(code)
        if len(d): assets[name]=d.set_index('date')['close'].astype(float).pct_change()
    w('assets '+', '.join(f'{k}:{len(v)}' for k,v in assets.items()))
    res={}
    for name,ret in assets.items():
        df=pd.DataFrame({'ret':ret})
        df['phase']=df.index.to_series().map(lambda d: clock['phase'].asof(pd.Timestamp(d).replace(day=1)))
        g=df.dropna().groupby('phase')['ret'].agg(['count','mean','std'])
        g['ann']=g['mean']*252; g['sharpe']=g['mean']/g['std']*np.sqrt(252)
        res[name]=g.round(4).to_dict('index')
    json.dump(res, open(OUT / 'clock_assets.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps({k:{p:{kk:round(vv,4) for kk,vv in v.items()} for p,v in d.items()} for k,d in res.items()}, ensure_ascii=False, indent=1))
    w('DONE')
except Exception:
    w(traceback.format_exc())
