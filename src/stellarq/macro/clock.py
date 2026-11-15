"""美林投资时钟：历史全量重建 + 时点对齐（point-in-time）。

与 ``insight-hub/app/macro_clock.py``（只取最近若干期、用于展示当前状态）不同，
本模块要能**回测**：每一期都必须只用当时已经公布的宏观数据。

数据源：东方财富数据中心（国家统计局口径）
  * ``RPT_ECONOMY_CPI``  月度 CPI 同比（NATIONAL_SAME）
  * ``RPT_ECONOMY_GDP``  季度 GDP 累计同比（SUM_SAME）
  * ``RPT_ECONOMY_PMI``  月度制造业 PMI（MAKE_INDEX）
  * ``RPT_ECONOMY_PPI``  月度 PPI 同比（BASE_SAME）

口径与坑
--------
1. **REPORT_DATE 是「统计期末」**（如 CPI 2024-03 记作 2024-03-01），不是公布日。
   真实公布日约在该月末 +15 天（CPI/PPI/PMI）或季后 +15~20 天（GDP）。
   因此这里统一滞后 ``RELEASE_LAG_DAYS`` 天，宁可保守也不允许前视。
2. **GDP 是累计同比**：4.6% 表示年初至今累计，不是单季。判方向用累计同比的差分，
   这在数学上等于是「本期单季增速 vs 上期单季增速」的近似，稳健但不精确。
3. PMI 与 PPI 用于**辅助确认**：PMI 在 50 上下、PPI 方向，不与 CPI/GDP 双向判据冲突时给加成。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
DC = "https://datacenter-web.eastmoney.com/api/data/v1/get"

# 统计期末 → 可知日期 的保守滞后（自然日）
RELEASE_LAG = {"cpi": 20, "ppi": 20, "pmi": 5, "gdp": 25}

PHASES = {
    ("up", "up"): ("过热", "增长↑ 通胀↑", "大宗商品 > 股票 > 债券"),
    ("up", "down"): ("复苏", "增长↑ 通胀↓", "股票 > 债券 > 商品"),
    ("down", "up"): ("滞胀", "增长↓ 通胀↑", "现金/商品抗跌，股债承压"),
    ("down", "down"): ("衰退", "增长↓ 通胀↓", "债券 > 现金 > 股票"),
}


def _fetch(report: str, timeout: float = 30.0) -> pd.DataFrame:
    rows, page = [], 1
    while True:
        for attempt in range(3):
            try:
                j = requests.get(DC, params={
                    "reportName": report, "columns": "ALL", "pageNumber": page,
                    "pageSize": 500, "sortColumns": "REPORT_DATE", "sortTypes": "1",
                }, headers=HEADERS, timeout=timeout).json()
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.5)
                j = {}
        res = j.get("result") or {}
        data = res.get("data") or []
        rows.extend(data)
        if page >= int(res.get("pages") or 1) or not data:
            break
        page += 1
    return pd.DataFrame(rows)


def macro_panel(start: str = "2012-01-01") -> pd.DataFrame:
    """月度宏观面板：统计期末索引，另存 ``known``（可知日期）列。"""
    cpi = _fetch("RPT_ECONOMY_CPI")
    gdp = _fetch("RPT_ECONOMY_GDP")
    pmi = _fetch("RPT_ECONOMY_PMI")
    ppi = _fetch("RPT_ECONOMY_PPI")

    def prep(df, col, name, lag_key):
        if not len(df):
            return pd.DataFrame(columns=[name, "known"])
        out = pd.DataFrame({
            "period_end": pd.to_datetime(df["REPORT_DATE"]),
            name: pd.to_numeric(df[col], errors="coerce"),
        }).dropna()
        out["known"] = out["period_end"] + pd.Timedelta(days=RELEASE_LAG[lag_key])
        return out.set_index("period_end")

    c = prep(cpi, "NATIONAL_SAME", "cpi_yoy", "cpi")
    p = prep(ppi, "BASE_SAME", "ppi_yoy", "ppi")
    m = prep(pmi, "MAKE_INDEX", "pmi", "pmi")
    g = prep(gdp, "SUM_SAME", "gdp_yoy", "gdp")

    idx = pd.date_range(start, pd.Timestamp.today().normalize(), freq="MS")
    panel = pd.DataFrame(index=idx)
    panel = panel.join(c[["cpi_yoy", "known"]].rename(columns={"known": "cpi_known"}))
    panel = panel.join(p[["ppi_yoy", "known"]].rename(columns={"known": "ppi_known"}))
    panel = panel.join(m[["pmi", "known"]].rename(columns={"known": "pmi_known"}))
    # GDP 是季度：向下填充到月（保留「最新已知」的累计同比）
    gq = g[["gdp_yoy", "known"]].rename(columns={"known": "gdp_known"})
    gq = gq[~gq.index.duplicated(keep="last")]
    panel = panel.join(gq.reindex(gq.index, method=None), how="left")
    panel[["gdp_yoy", "gdp_known"]] = gq.reindex(panel.index, method="ffill").to_numpy()
    panel = panel.ffill()
    return panel


def classify(panel: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """按「增长方向 × 通胀方向」划四象限。

    方向判定用最近 ``lookback`` 期的变化（差分），不依赖单期噪声。
    """
    out = panel.copy()
    # 直接对 numpy 数组做差分：本环境 pandas 3 下 np.sign(Series) 会报 unorderable types
    for col, new in (("gdp_yoy", "gdp_dir"), ("cpi_yoy", "cpi_dir"),
                     ("ppi_yoy", "ppi_dir"), ("pmi", "pmi_dir")):
        arr = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
        d = np.full(len(arr), np.nan)
        d[lookback:] = arr[lookback:] - arr[:-lookback]
        out[new] = np.where(np.isfinite(d), np.sign(d), np.nan)

    def phase(r):
        g, c = r["gdp_dir"], r["cpi_dir"]
        if not np.isfinite(g) or not np.isfinite(c) or g == 0 or c == 0:
            return "无法判定"
        key = ("up" if g > 0 else "down", "up" if c > 0 else "down")
        return PHASES[key][0]

    out["phase"] = out.apply(phase, axis=1)
    # 辅助确认：PMI>50 且方向向上 → 增长更强的证据
    out["pmi_confirm"] = np.where(out["pmi"] >= 50, "扩张", "收缩")
    return out


def phase_for_dates(clock: pd.DataFrame, dates) -> np.ndarray:
    """把月度相位映射到交易日（用可知日期对齐，严禁前视）。"""
    known = clock["cpi_known"].to_numpy()
    order = np.argsort(known)
    known_sorted = known[order]
    phases = clock["phase"].to_numpy()[order]
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    pos = np.searchsorted(known_sorted, d, side="right") - 1
    res = np.where(pos >= 0, phases[np.clip(pos, 0, None)], "无法判定")
    return res.astype(object)
