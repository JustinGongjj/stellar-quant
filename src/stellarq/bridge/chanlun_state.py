"""缠论结构状态序列：把「一次性结构快照」变成「逐根因果状态」。

为什么需要这一层
----------------
上一轮接入（reports/缠论技术面接入.md）的结论是：缠论结构**当截面因子**无效
（修正前视偏差后 IC −0.042 ~ +0.006）。但那份报告同时留了三个未否掉的疑点：

  1. 只有 4 个截面，统计力弱；
  2. **买卖点可能是择时信号而非截面信号**——它回答「这只现在能不能买」，
     不是「这批里哪只更好」。测的是后者；
  3. **多级别联立（区间套）没测**。

本模块解决的是这些疑点的共同前提：**必须先有逐根的结构状态**。
旧桥接 `bridge/chanlun.py` 只取"当前"一个快照（无 date 维度），
既无法接进 walk-forward 面板，也天然带前视偏差——它就是那个 bug 的来源。

因果性纪律（本模块的核心约束）
----------------------------
`/api/chan` 是在**整段 K 线**上一次算完的。直接取 `trend.type` 当"当天状态"是错的：
它用到了 t 之后的数据。本模块的投影规则是：

  **在 bar t 上，只使用 `i1 <= t` 的结构元素。**

- 笔：`bi.i1 <= t` 才算"已成笔"
- 中枢：`pivot.i1 <= t` 才算"已收口中枢"；未收口的（`status=='ongoing'`）
  只有在 `i0 <= t` 时以"进行中中枢"身份参与，且单独标记，不混入已确认计数
- 信号/背驰：`barIdx <= t`，并给出 `age = t - barIdx`
- 走势类型：由已确认中枢的关系**在本地重算**，不取 API 的 `trend.type`

这样得到的每一行，与"只用 t 及之前数据重算"等价（就结构元素可见性而言）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 与缠论工作站一致的编码
TREND_CODE = {"上涨趋势": 1, "下跌趋势": -1, "盘整": 0}
SIGNAL_SIDE = {"1B": 1, "2B": 1, "3B": 1, "1S": -1, "2S": -1, "3S": -1}


def _trend_from_pivots(pivots: list[dict]) -> tuple[int, int]:
    """由已确认中枢的相邻关系判定走势，返回 (方向, 同向连续个数)。

    口径与缠论工作站 classifyTrend 一致：相邻中枢 zg/zd 不重叠才算方向，
    重叠即打断连续。
    """
    if not pivots:
        return 0, 0
    if len(pivots) == 1:
        return 0, 0
    rels = []
    for i in range(1, len(pivots)):
        a, b = pivots[i - 1], pivots[i]
        if b["zd"] > a["zg"]:
            rels.append(1)
        elif b["zg"] < a["zd"]:
            rels.append(-1)
        else:
            rels.append(0)
    direction, run = 0, 0
    for r in reversed(rels):
        if r == 0:
            break
        if direction == 0:
            direction, run = r, 1
        elif r == direction:
            run += 1
        else:
            break
    return direction, run


def state_series(chan_payload: dict) -> pd.DataFrame:
    """把 /api/chan 的返回投影成逐根结构状态。

    返回 DataFrame，行 = bar（按时间升序），列见文件末 STATE_COLUMNS。
    `date` 列来自 bars[].t（毫秒时间戳转日期）。
    """
    bars = chan_payload.get("bars") or []
    if not bars:
        return pd.DataFrame(columns=["date"] + STATE_COLUMNS)

    n = len(bars)
    dates = pd.to_datetime([b["t"] for b in bars], unit="ms").normalize()
    close = np.asarray([b["c"] for b in bars], dtype=float)
    high = np.asarray([b["h"] for b in bars], dtype=float)
    low = np.asarray([b["l"] for b in bars], dtype=float)

    bis = sorted(chan_payload.get("bis") or [], key=lambda x: x["i1"])
    pivots = sorted(chan_payload.get("pivots") or [], key=lambda x: x["i1"])
    signals = chan_payload.get("signals") or []
    beichi = chan_payload.get("beichi") or []

    macd = chan_payload.get("macd") or {}
    hist = np.asarray(macd.get("hist") or [np.nan] * n, dtype=float)

    # 按 barIdx 建索引，便于在 t 上 O(1) 查"是否有信号/背驰"
    sig_by_bar: dict[int, dict] = {}
    for s in signals:
        bi = s.get("barIdx")
        if bi is None:
            continue
        bi = int(bi)
        prev = sig_by_bar.get(bi)
        # 同根多信号取"最新写入的"（API 已按时间排序，后者更新）
        sig_by_bar[bi] = s
    bei_by_bar: dict[int, dict] = {}
    for b in beichi:
        bi = b.get("barIdx")
        if bi is None:
            continue
        bei_by_bar[int(bi)] = b

    rows = []
    bi_ptr, pv_ptr = 0, 0
    confirmed_pivots: list[dict] = []
    last_bi = None
    ongoing_pivot = None
    last_signal = None
    last_signal_bar = None
    last_beichi = None
    last_beichi_bar = None

    for t in range(n):
        # --- 推进已确认笔（i1 <= t） ---
        while bi_ptr < len(bis) and bis[bi_ptr]["i1"] <= t:
            last_bi = bis[bi_ptr]
            bi_ptr += 1

        # --- 推进已收口中枢（i1 <= t） ---
        while pv_ptr < len(pivots) and pivots[pv_ptr]["i1"] <= t:
            confirmed_pivots.append(pivots[pv_ptr])
            pv_ptr += 1

        # --- 进行中的中枢：i0 <= t < i1，单独标记，不进 confirmed ---
        ongoing_pivot = None
        for p in pivots:
            if p["i0"] <= t < p["i1"]:
                ongoing_pivot = p
                break

        # --- 信号 / 背驰（barIdx <= t） ---
        if t in sig_by_bar:
            last_signal = sig_by_bar[t]
            last_signal_bar = t
        if t in bei_by_bar:
            last_beichi = bei_by_bar[t]
            last_beichi_bar = t

        # --- 中枢位置（以最近一个已确认中枢为参照） ---
        ref = confirmed_pivots[-1] if confirmed_pivots else None
        pos_code, pos_pct, pivot_h = 0.0, np.nan, np.nan
        if ref:
            zg, zd = float(ref["zg"]), float(ref["zd"])
            h = max(zg - zd, 1e-9)
            pivot_h = h
            mid = zd + h / 2
            pos_pct = (close[t] - mid) / h
            if close[t] > zg:
                pos_code = 1.0
            elif close[t] < zd:
                pos_code = -1.0
            else:
                pos_code = 0.0

        # --- 走势类型（本地重算，不看 API 的 trend） ---
        tdir, trun = _trend_from_pivots(confirmed_pivots)

        # --- 笔的方向/长度/力度 ---
        bi_dir = int(last_bi["dir"]) if last_bi else 0
        bi_len = (int(last_bi["i1"]) - int(last_bi["i0"]) + 1) if last_bi else np.nan
        bi_age = (t - int(last_bi["i1"])) if last_bi else np.nan

        # 该笔的 MACD 面积（力度），用 hist 绝对值和
        bi_area = np.nan
        if last_bi:
            a, b = int(last_bi["i0"]), int(last_bi["i1"])
            seg = hist[a : b + 1]
            if seg.size and np.isfinite(seg).any():
                bi_area = float(np.nansum(np.abs(seg)))

        rec = {
            "date": dates[t],
            # 结构计数
            "cs_pivots": len(confirmed_pivots),
            "cs_trend": tdir,
            "cs_trend_run": trun,
            # 位置
            "cs_pos": pos_code,
            "cs_pos_pct": pos_pct,
            "cs_pivot_h": pivot_h,
            "cs_pivot_w": (pivot_h / close[t]) if (ref and close[t] > 0) else np.nan,
            # 中枢年龄：距最近中枢收口多少根
            "cs_pivot_age": (t - int(ref["i1"])) if ref else np.nan,
            # 笔
            "cs_bi_dir": bi_dir,
            "cs_bi_len": bi_len,
            "cs_bi_age": bi_age,
            "cs_bi_area": bi_area,
            # 信号
            "cs_sig": (SIGNAL_SIDE.get(str((last_signal or {}).get("type")), 0)
                       if last_signal else 0),
            "cs_sig_age": (t - last_signal_bar) if last_signal_bar is not None else np.nan,
            "cs_sig_kind": str((last_signal or {}).get("type") or ""),
            # 背驰
            "cs_bei": 1 if last_beichi else 0,
            "cs_bei_ratio": (float(last_beichi.get("areaRatio"))
                             if last_beichi and last_beichi.get("areaRatio") is not None
                             else np.nan),
            "cs_bei_age": (t - last_beichi_bar) if last_beichi_bar is not None else np.nan,
            # 进行中中枢（未收口）
            "cs_ongoing": 1 if ongoing_pivot is not None else 0,
        }
        rows.append(rec)

    return pd.DataFrame(rows)


STATE_COLUMNS = [
    "cs_pivots", "cs_trend", "cs_trend_run",
    "cs_pos", "cs_pos_pct", "cs_pivot_h", "cs_pivot_w", "cs_pivot_age",
    "cs_bi_dir", "cs_bi_len", "cs_bi_age", "cs_bi_area",
    "cs_sig", "cs_sig_age", "cs_sig_kind",
    "cs_bei", "cs_bei_ratio", "cs_bei_age",
    "cs_ongoing",
]

#: 数值型状态列（可直接进矩阵）
NUMERIC_STATE = [c for c in STATE_COLUMNS if c != "cs_sig_kind"]
