"""对冲效果评估：用真实指数期货的代理（指数本身）做 beta 中性，看能否把 IR 转成绝对收益。
from stellarq.paths import DATA, ROOT  # noqa: E402

数据限制（必须说明）：
  腾讯接口不提供股指期货行情（IF0/IC0/IM0 各种写法都返回 param error），
  因此这里用**沪深300 / 中证500 / 中证1000 指数**做对冲标的的**代理**。
  真实期货有基差（贴水），A 股近月合约年化贴水约 -3% ~ -8%，会侵蚀对冲收益；
  本脚本额外做一个「贴水 -5%」的敏感性，展示真实成本。

对冲方式：每期用组合 60 日的 beta 估算对冲比例，空头仓位在指数上取得 -beta×指数收益。
"""
import sys, json, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    from stellarq.data.tencent import Tencent
    t=Tencent()
    idx={'hs300':t.daily('399300'),'zz500':t.daily('000905'),'zz1000':t.daily('000852')}
    for k,v in idx.items(): w(f'{k} {len(v)} {v["date"].min()} -> {v["date"].max()}')
    C = str(__import__('pathlib').Path.home() / '.cache/stellarq')
    pf=pd.read_csv(OUT / 'opt2_B.csv', parse_dates=['date'])
    # 组合逐期净收益 → 累成日频并不现实（5 日一期），改用「期收益 减去 beta×同期指数收益」
    res={}
    for name,df_idx in idx.items():
        s=df_idx.set_index('date')['close'].astype(float)
        ret=s.pct_change()
        rows=[]
        for _,r in pf.iterrows():
            d0=pd.Timestamp(r['date'])
            # 持仓期：T+1 开盘买 → T+6 开盘卖，用指数 T+1 到 T+6 的收益近似
            sub=ret[(ret.index>d0)]
            if len(sub)<6: continue
            idx_ret=float((1+sub.iloc[1:6]).prod()-1)
            rows.append({'date':r['date'],'net':r['net'],'turn':r['turn'],'idx':idx_ret})
        m=pd.DataFrame(rows)
        if len(m)<30: continue
        # 滚动 60 期 beta
        beta=m['net'].rolling(20,min_periods=10).cov(m['idx'])/m['idx'].rolling(20,min_periods=10).var()
        beta=beta.bfill().clip(-1.5,1.5)
        hedged=m['net']-beta*m['idx']
        ov=hedged.rolling(5).mean().dropna()
        nav=(1+ov).cumprod()
        # 贴水成本：期货空头年化 -5% 贴水，按持有期折算
        basis=-0.05*5/252
        ov_b=ov+basis
        navb=(1+ov_b).cumprod()
        res[name]={'n':int(len(m)),'beta_mean':float(beta.mean()),
                   'unhedged_ann':float(m['net'].rolling(5).mean().dropna().mean()*10.08),
                   'hedged_ann':float(ov.mean()*10.08),
                   'hedged_sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08)),
                   'hedged_ann_with_basis':float(ov_b.mean()*10.08),
                   'hedged_sharpe_with_basis':float(ov_b.mean()/ov_b.std(ddof=1)*np.sqrt(10.08)),
                   'hedged_max_dd':float((nav/nav.cummax()-1).min()),
                   'hedged_max_dd_with_basis':float((navb/navb.cummax()-1).min())}
        w(json.dumps({name:res[name]}, ensure_ascii=False))
    json.dump(res, open(OUT / 'hedge.json','w'), ensure_ascii=False, indent=1, default=float)
    w('DONE')
except Exception:
    w(traceback.format_exc())
