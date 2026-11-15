"""华泰 AI 系列 10 的方法做出来：宏观状态变量直接进模型 + 短滚动训练窗口。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

与前面几轮的区别（这是关键）：
  * 之前我把「时钟相位」当规则去做择时/过滤 → 失败（phase11/13）；
  * 华泰的做法是**把宏观变量作为特征喂给树模型**，让模型自己学「什么状态下该信哪个因子」；
  * 同时训练窗口缩短到 6 个月（报告明确结论：长训练期在风格切换时回撤大）。

特征 = 19 个量价 + 6 个宏观状态变量（全部按可知日期滞后，无前视）：
  cpi_yoy, cpi_mom(3期差分), gdp_yoy, gdp_mom, pmi, pmi_dev(PMI-50)
训练：滚动 6 个月窗口，每 5 个交易日重训一次。
对照：同结构但去掉宏观变量（纯量价）。
"""
import sys, json, glob, resource, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m): print(f'{m} [mem {round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}MB]', flush=True)
try:
    import numpy as np, pandas as pd
    from stellarq.macro.clock import macro_panel
    from stellarq.models.gbdt import GBDTParams, HistGBDT
    C='CACHE/'; R=C+'wide2/'
    X=np.load(C+'v4_X.npy'); y=np.load(C+'v4_y.npy'); d=np.load(C+'v4_d.npy'); sidx=np.load(C+'v4_s.npy')
    dates=np.load(R+'wide_dates.npy',allow_pickle=True)
    V4=['KMID','LN_CAP','MA10','MA20','MA60','MA20_resid','MA60_resid','STD20','MIN5','MIN20','RET20','TURN5',
        'VSTD20','VSTD20_resid','AMT_RATIO60','AMT_RATIO60_resid','VMA60','VSUMD60','CORR20']
    # 宏观状态变量（按可知日期）
    panel=macro_panel('2012-01-01')
    mk=pd.DataFrame({
        'cpi_yoy':panel['cpi_yoy'],
        'cpi_mom':panel['cpi_yoy'].diff(3),
        'gdp_yoy':panel['gdp_yoy'],
        'gdp_mom':panel['gdp_yoy'].diff(3),
        'pmi':panel['pmi'],
        'pmi_dev':panel['pmi']-50.0,
    })
    known=pd.to_datetime(panel['cpi_known']).to_numpy()
    order=np.argsort(known)
    mk=mk.iloc[order]; known=known[order]
    MACRO_COLS=list(mk.columns)
    # 给每个样本行分配「其日期之前最新可知」的宏观向量
    uniq=np.unique(d)                               # 训练网格的日期序号
    dt=pd.to_datetime(pd.Series(dates)).to_numpy()
    mac_at={}
    for u in uniq:
        i=np.searchsorted(known, dt[u], side='right')-1
        mac_at[u]=mk.iloc[max(i,0)].to_numpy(dtype=np.float32) if i>=0 else np.zeros(len(MACRO_COLS),np.float32)
    M=np.vstack([mac_at[u] for u in d]).astype(np.float32)
    w(f'macro matrix {M.shape}')
    XM=np.hstack([X, M]).astype(np.float32)
    FEATS_M=V4+MACRO_COLS
    FEATS_P=V4
    # 滚动 6 个月（约 120 个交易日 ≈ 24 个 5 日网格点）重训
    grid=sorted(uniq.tolist())
    cut=grid[int(len(grid)*0.85)]
    test_idx=np.where(d>=cut)[0]
    test_dates=sorted(set(d[d>=cut].tolist()))
    WINDOW=24
    def rolling_predict(Xa, feats):
        pred=np.full(len(test_idx), np.nan)
        WINDOW_SLICE=WINDOW
        for k,td in enumerate(test_dates):
            gpos=grid.index(td)
            lo=grid[max(0,gpos-WINDOW_SLICE)]
            tr_mask=(d>=lo)&(d<td)
            if tr_mask.sum()<2000: continue
            mdl=HistGBDT(GBDTParams(n_estimators=40,max_depth=4,min_child_samples=200,subsample=0.7,colsample=0.8))
            mdl.fit(Xa[tr_mask].astype(np.float64), y[tr_mask].astype(np.float64), feats)
            sel=np.where(d==td)[0]
            pred[np.isin(test_idx, sel)]=mdl.predict(Xa[sel].astype(np.float64))
        return pred
    def ic_of(pred):
        out=[]
        for td in test_dates:
            sel=np.where(d==td)[0]
            p=pred[np.isin(test_idx,sel)]; yy=y[sel]
            ok=np.isfinite(p)&np.isfinite(yy)
            if ok.sum()<100: continue
            ra=np.argsort(np.argsort(p[ok],kind='mergesort')).astype(float)
            rb=np.argsort(np.argsort(yy[ok],kind='mergesort')).astype(float)
            if ra.std()>0 and rb.std()>0:
                out.append(float(((ra-ra.mean())*(rb-rb.mean())).mean()/(ra.std()*rb.std())))
        return np.array(out)
    res={}
    for name,Xa,feats in (('macro_6m',XM,FEATS_M),('price_only_6m',X,FEATS_P)):
        p=rolling_predict(Xa,feats)
        ics=ic_of(p)
        res[name]={'n_dates':len(test_dates),'ic_mean':float(ics.mean()),'icir':float(ics.mean()/ics.std(ddof=1)),
                   'ic_win':float((ics>0).mean())}
        w(f'{name}: IC {res[name]["ic_mean"]:+.4f} ICIR {res[name]["icir"]:+.3f} 胜率 {res[name]["ic_win"]:.2%}')
    json.dump(res, open(OUT / 'macro_state_model.json','w'), ensure_ascii=False, indent=1, default=float)
    w('DONE')
except Exception:
    w(traceback.format_exc())
