"""事件因子与基本面因子（无前视口径）。

两条铁律：
1. **时点用公告日，不用报告期**。基本面在 ``NOTICE_DATE`` 之前对市场不可见；
2. **同一天只用当天收盘前已知的信息**，因此因子在 t 日的值 = 公告日 ≤ t 的记录。

基本面快照的「修正」与「重复披露」：同一 (代码, 报告期) 会有多条记录（不同披露时点），
取 ``NOTICE_DATE`` 最新且 ≤ t 的那一条，避免用到后来的修订值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLS = [
    "ev_forecast_20d", "ev_express_20d", "ev_buyback_60d", "ev_increase_60d",
    "ev_reduce_60d", "ev_restructure_120d", "ev_penalty_120d", "ev_incentive_120d",
    "ev_abnormal_20d",
]
EVENT_WINDOW = {
    "forecast": 20, "express": 20, "buyback": 60, "increase": 60, "reduce": 60,
    "restructure": 120, "penalty": 120, "equity_incentive": 120, "abnormal": 20,
}


def event_matrix(events: pd.DataFrame, dates: np.ndarray, symbols: np.ndarray,
                 out_dir: str | None = None) -> dict[str, np.ndarray]:
    """把公告事件聚合到 (date × symbol) 的计数矩阵。

    ``ev_<cat>_<W>d`` = 过去 W 个交易日内公告次数（含当日）。

    内存：本环境 cgroup 2 GB，9 个 (3581×5027) 矩阵 = 648 MB，加上 pandas 的开销必然 OOM。
    因此**一次只算一个类别并立即返回**（生成器语义），由调用方逐个使用/落盘。
    """
    dmap = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    smap = {str(s): i for i, s in enumerate(symbols)}
    nd, ns = len(dates), len(symbols)
    ev = events.copy()
    ev["ann_date"] = pd.to_datetime(ev["ann_date"]).dt.normalize()
    ev["di"] = ev["ann_date"].map(dmap)
    ev["si"] = ev["symbol"].astype(str).map(smap)
    ev = ev.dropna(subset=["di", "si"])
    out = {}
    for cat, win in EVENT_WINDOW.items():
        sub = ev[ev["event"] == cat]
        raw = np.zeros((nd, ns), dtype=np.float32)
        if len(sub):
            np.add.at(raw, (sub["di"].astype(int).to_numpy(), sub["si"].astype(int).to_numpy()), 1.0)
        cs = np.cumsum(raw, axis=0)
        res = np.empty_like(cs)
        res[:win] = cs[:win]
        res[win:] = cs[win:] - cs[:-win]
        out[cat] = res
    return out


def fundamental_panel(fund_files: list[str], cutoff_date) -> pd.DataFrame:
    """合并季度财务，保留 (code, 报告期, 公告日, 关键指标)。"""
    frames = []
    for f in fund_files:
        try:
            d = pd.read_parquet(f)
        except Exception:  # noqa: BLE001
            continue
        if len(d):
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["SECURITY_CODE", "NOTICE_DATE"])
    df["symbol"] = df["SECURITY_CODE"].astype(str).str.zfill(6)
    df["notice"] = pd.to_datetime(df["NOTICE_DATE"]).dt.normalize()
    df["report"] = pd.to_datetime(df["REPORTDATE"]).dt.normalize()
    df = df[df["notice"] <= pd.Timestamp(cutoff_date)]
    num = ["TOTAL_OPERATE_INCOME", "PARENT_NETPROFIT", "YSTZ", "SJLTZ", "WEIGHTAVG_ROE",
           "XSMLL", "BASIC_EPS", "BPS", "MGJYXJJE", "SJLHZ"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["symbol", "report", "notice"] + num].drop_duplicates(["symbol", "report", "notice"])


def point_in_time(df: pd.DataFrame, dates: np.ndarray, symbols: np.ndarray,
                  fields: list[str], chunk: int = 400) -> dict[str, np.ndarray]:
    """把「公告日快照」变成逐日可用矩阵（只用 ≤ t 的最新一条，杜绝前视）。

    实现：对每条记录算出它「生效的起始日序号」，一次性 scatter 到逐日矩阵，
    再沿时间轴做 forward-fill（用 maximum.accumulate 的索引技巧）。
    全程只分配 (日期 × 股票) 的 float32 矩阵，峰值内存可控。
    """
    smap = pd.Series(np.arange(len(symbols), dtype=np.int32),
                     index=pd.Index([str(s) for s in symbols]))
    si = df["symbol"].astype(str).map(smap)
    keep = si.notna()
    si = si[keep].astype(np.int64).to_numpy()
    notices = pd.to_datetime(df.loc[keep, "notice"]).to_numpy()
    dts = pd.to_datetime(pd.Series(dates)).to_numpy()
    # 公告日 → 生效起始的日序号（公告日当天即可用；非交易日则从下一个交易日起生效）
    start = np.searchsorted(dts, notices, side="left")
    nd, ns = len(dates), len(symbols)
    out = {}
    for f in fields:
        vals = df.loc[keep, f].to_numpy(dtype=np.float64)
        good = np.isfinite(vals) & (start < nd)
        s_i, si_i, v_i = start[good], si[good], vals[good]
        # 同一 (日期, 股票) 多条时取「公告日最晚」的值：按公告日排序后，后写覆盖
        order = np.lexsort((notices[good], s_i))
        m = np.full((nd, ns), np.nan, dtype=np.float32)
        m[s_i[order], si_i[order]] = v_i[order].astype(np.float32)
        # 逐列 forward-fill（分块，避免大索引数组）
        filled = np.empty_like(m)
        for j0 in range(0, nd, chunk):
            j1 = min(j0 + chunk, nd)
            block = m[j0:j1]
            idx = np.where(np.isfinite(block), np.arange(j1 - j0)[:, None],
                           -np.ones((j1 - j0, 1), dtype=np.int64))
            np.maximum.accumulate(idx, axis=0, out=idx)
            res = np.take_along_axis(block, np.clip(idx, 0, None), axis=0)
            res[idx < 0] = np.nan
            filled[j0:j1] = res
        out[f] = filled
    return out
