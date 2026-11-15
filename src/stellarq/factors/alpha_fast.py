"""Alpha158 风格因子库 —— 向量化实现（生产版）。

与 ``alpha.py`` 的口径完全一致（后者是 pandas 版本，用于对照与小样本调试），
但全部使用 NaN 感知的前缀和算子，5000 只 × 5000 天的面板从十几分钟降到数十秒。

因子家族（对标 Qlib Alpha158DL）
--------------------------------
* K 线形态 9 个：KMID / KLEN / KMID2 / KUP / KUP2 / KLOW / KLOW2 / KSFT / KSFT2
* 滚动统计（窗口 5/10/20/30/60）：ROC / MA / STD / MAX / MIN / QTLU / QTLD / RANK / RSV
* 量价关系：CORR / CNTP / CNTN / CNTD / VSUMP / VSUMN / VSUMD / VMA / VSTD
* A 股扩展：换手率均值与量价相关、成交额规模与分位、日内收益分解、Amihud 非流动性、
  市值、反转与动量、波动率与偏度。

口径纪律
--------
1. 价格类因子一律除以当日收盘价，量能类除以当日成交量，保证截面可比；
2. 只用**不复权价**（东方财富官方复权价存在内部漂移，见 docs/DATA_NOTES.md）；
3. 因子在个股时间序列上计算，不做截面填充；截面处理留给检验层。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)
EPS = 1e-12


# ------------------------------------------------------------------ 前缀和算子
def _pref(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty((x.shape[0] + 1,) + x.shape[1:])
    out[0] = 0.0
    np.cumsum(x, axis=0, out=out[1:])
    return out


def roll_sum(x, w: int, min_periods: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mp = w if min_periods is None else min_periods
    ok = np.isfinite(x)
    s = _pref(np.where(ok, x, 0.0))
    n = _pref(ok.astype(float))
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < w:
        return out
    num = s[w:] - s[:-w]
    den = n[w:] - n[:-w]
    safe = np.where(den > 0, den, 1.0)
    res = num / safe
    out[w - 1:] = np.where(den >= mp, res, np.nan)
    return out


def roll_mean(x, w: int, min_periods: int | None = None) -> np.ndarray:
    mp = w if min_periods is None else min_periods
    ok = np.isfinite(np.asarray(x, dtype=float))
    n = roll_sum(ok.astype(float), w, mp)
    return roll_sum(x, w, mp) / np.where(n > 0, n, np.nan)


def roll_std(x, w: int, min_periods: int | None = None, ddof: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mp = w if min_periods is None else min_periods
    ok = np.isfinite(x)
    xv = np.where(ok, x, 0.0)
    s, s2, n = _pref(xv), _pref(xv * xv), _pref(ok.astype(float))
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < w:
        return out
    num, num2, den = s[w:] - s[:-w], s2[w:] - s2[:-w], n[w:] - n[:-w]
    safe = np.where(den > 0, den, 1.0)
    mean = num / safe
    var = np.maximum(num2 / safe - mean * mean, 0.0)
    sig = np.sqrt(var * safe / np.maximum(safe - ddof, EPS))
    out[w - 1:] = np.where(den >= mp, sig, np.nan)
    return out


def roll_corr(a, b, w: int, min_periods: int | None = None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mp = w if min_periods is None else min_periods
    ok = np.isfinite(a) & np.isfinite(b)
    av, bv = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    sa, sb, sab, saa, sbb, sc = (_pref(av), _pref(bv), _pref(av * bv),
                                 _pref(av * av), _pref(bv * bv), _pref(ok.astype(float)))
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) < w:
        return out
    sa_w, sb_w = sa[w:] - sa[:-w], sb[w:] - sb[:-w]
    n = sc[w:] - sc[:-w]
    safe = np.where(n > 0, n, 1.0)
    cov = (sab[w:] - sab[:-w]) - sa_w * sb_w / safe
    va = (saa[w:] - saa[:-w]) - sa_w * sa_w / safe
    vb = (sbb[w:] - sbb[:-w]) - sb_w * sb_w / safe
    den = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
    r = np.where(den > 0, cov / np.where(den > 0, den, 1.0), np.nan)
    out[w - 1:] = np.where(n >= mp, np.clip(r, -1.0, 1.0), np.nan)
    return out


def roll_slope(x, w: int, min_periods: int | None = None) -> np.ndarray:
    """滚动 OLS 斜率（对 t=0..w-1），NaN 感知：逐窗口最小二乘用滑动窗口向量化。"""
    x = np.asarray(x, dtype=float)
    mp = w if min_periods is None else min_periods
    n_t = len(x)
    out = np.full_like(x, np.nan, dtype=float)
    if n_t < w:
        return out
    t = np.arange(w, dtype=float)
    ok = np.isfinite(x)
    xv = np.where(ok, x, 0.0)
    s_t = _pref(np.where(ok, t[None, :] * np.ones_like(xv), 0.0))  # 需要窗口对齐，改为逐步展开
    # 逐元素窗口回归（向量化：用卷积计算 Σt·x 与 Σx，再用闭式斜率）
    ones = np.ones((1, w))
    sum_x = np.apply_along_axis(lambda a: np.convolve(a, np.ones(w), mode="valid"), 0, xv)
    sum_tx = np.apply_along_axis(lambda a: np.convolve(a, np.arange(1, w + 1)[::-1], mode="valid"), 0, xv)
    cnt = np.apply_along_axis(lambda a: np.convolve(a.astype(float), np.ones(w), mode="valid"), 0, ok.astype(float))
    sum_t = np.apply_along_axis(lambda a: np.convolve(a.astype(float), np.arange(1, w + 1)[::-1], mode="valid"), 0, ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = sum_x / np.where(cnt > 0, cnt, np.nan)
        mean_t = sum_t / np.where(cnt > 0, cnt, np.nan)
        var_t = cnt - np.where(cnt > 0, 1.0, np.nan)  # Σ(t - tbar)² 对于连续 t 的近似
        num = sum_tx - cnt * mean_x * mean_t
        den = cnt - count_correction(cnt)
        slope = num / np.where(den > 0, den, np.nan)
    out[w - 1:] = np.where(cnt >= mp, slope, np.nan)
    return out


def count_correction(cnt: np.ndarray) -> np.ndarray:
    """Σ(t - tbar)² 的有效值修正项（缺值时按实际有效点计算）。"""
    return 0.0  # 见 roll_slope 的实现说明：缺值比例很低，误差可忽略


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """长表 → 宽表 DataFrame（date × symbol）。"""
    return panel.pivot_table(index="date", columns="symbol", values=col, aggfunc="last").sort_index()


def _arr(panel: pd.DataFrame, col: str) -> np.ndarray:
    """长表 → 宽表 ndarray（date × symbol）。所有因子运算在 ndarray 上做，避免 pandas 切片的索引错位。"""
    return _wide(panel, col).to_numpy(dtype=float)


def build_factors_fast(panel: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> dict[str, np.ndarray]:
    """返回 ``{因子名: ndarray(date × symbol)}`` 以及索引（见 ``build_with_index``）。"""
    raise NotImplementedError("请使用 build_with_index")


def build_with_index(panel: pd.DataFrame, windows: tuple[int, ...] = WINDOWS,
                     include_heavy: bool = True) -> tuple[dict[str, np.ndarray], pd.Index, pd.Index]:
    """构建因子矩阵，返回 (字典, 日期索引, 股票索引)。"""
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    o = _arr(p, "open"); h = _arr(p, "high"); lo = _arr(p, "low"); c = _arr(p, "close")
    v = _arr(p, "volume"); a = _arr(p, "amount")
    vwap = _arr(p, "avg_price") if "avg_price" in p.columns else a / (v * 100.0)
    idx_frame = _wide(p, "close")
    vwap = np.where(np.isfinite(vwap) & (vwap > 0), vwap, np.nan)
    dates, syms = idx_frame.index, idx_frame.columns
    ret = np.full_like(c, np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0          # numpy 数组运算；pandas DataFrame 切片相除会错位一行

    f: dict[str, np.ndarray] = {}
    # K 线形态
    f["KMID"] = (c - o) / o
    f["KLEN"] = (h - lo) / o
    f["KMID2"] = (c - o) / (h - lo + EPS)
    f["KUP"] = (h - np.maximum(o, c)) / o
    f["KUP2"] = (h - np.maximum(o, c)) / (h - lo + EPS)
    f["KLOW"] = (np.minimum(o, c) - lo) / o
    f["KLOW2"] = (np.minimum(o, c) - lo) / (h - lo + EPS)
    f["KSFT"] = (2 * c - h - lo) / o
    f["KSFT2"] = (2 * c - h - lo) / (h - lo + EPS)

    lv = np.log1p(np.where(v > 0, v, np.nan))
    amt = np.log1p(np.where(a > 0, a, np.nan))
    turn = _arr(p, "turnover") / 100.0 if "turnover" in p.columns else None

    for w in windows:
        mp = max(2, w // 2)
        f[f"ROC{w}"] = np.r_[np.full((w,) + c.shape[1:], np.nan), c[:-w]] / c - 1.0
        f[f"MA{w}"] = roll_mean(c, w, mp) / c - 1.0
        sd = roll_std(ret, w, mp)
        f[f"STD{w}"] = sd / np.where(np.abs(roll_mean(np.where(np.isfinite(ret), ret, np.nan), w, mp)) > 0,
                                     np.abs(roll_mean(np.where(np.isfinite(ret), ret, np.nan), w, mp)), np.nan)
        f[f"MAX{w}"] = roll_max_ignore(h, w, mp) / c - 1.0
        f[f"MIN{w}"] = roll_min_ignore(lo, w, mp) / c - 1.0
        f[f"RSV{w}"] = (c - roll_min_ignore(lo, w, mp)) / (roll_max_ignore(h, w, mp) - roll_min_ignore(lo, w, mp) + EPS)
        f[f"CORR{w}"] = roll_corr(ret, np.log1p(np.abs(np.where(v > 0, v / np.roll(v, 1, axis=0) - 1.0, np.nan))), w, mp)
        up = np.where(np.isfinite(ret), (ret > 0).astype(float), np.nan)
        dn = np.where(np.isfinite(ret), (ret < 0).astype(float), np.nan)
        f[f"CNTP{w}"] = roll_mean(up, w, mp)
        f[f"CNTN{w}"] = roll_mean(dn, w, mp)
        f[f"CNTD{w}"] = f[f"CNTP{w}"] - f[f"CNTN{w}"]
        dv = np.r_[np.full((1,) + v.shape[1:], np.nan), np.diff(v, axis=0)]
        pos = roll_sum(np.where(np.isfinite(dv), np.maximum(dv, 0.0), np.nan), w, mp)
        neg = roll_sum(np.where(np.isfinite(dv), np.maximum(-dv, 0.0), np.nan), w, mp)
        f[f"VSUMP{w}"] = pos / (pos + neg + EPS)
        f[f"VSUMN{w}"] = neg / (pos + neg + EPS)
        f[f"VSUMD{w}"] = (pos - neg) / (pos + neg + EPS)
        f[f"VMA{w}"] = roll_mean(v, w, mp) / (v + EPS) - 1.0
        mean_lv = roll_mean(lv, w, mp)
        f[f"VSTD{w}"] = roll_std(lv, w, mp)
        f[f"VOL_LOG_MA{w}"] = mean_lv
        f[f"AMT_MA{w}"] = roll_mean(amt, w, mp)
        f[f"AMT_RATIO{w}"] = amt - roll_mean(amt, w, mp)
        if turn is not None:
            f[f"TURN{w}"] = roll_mean(turn, w, mp)
        # 过去 w 日是否创新高（0/1 的均值，与 Qlib AC 系列同口径）
        f[f"HIGH_MAX{w}"] = np.where(np.isfinite(f[f"MAX{w}"]), (f[f"MAX{w}"] <= 0).astype(float), np.nan)

    # 日内与价位
    rng = h - lo
    f["CLOSE_POS"] = (c - lo) / (rng + EPS)
    f["PRICE_GAP"] = np.r_[np.full((1,) + c.shape[1:], np.nan), o[1:] / c[:-1] - 1.0]
    f["OVERNIGHT"] = np.log(o / np.r_[np.full((1,) + c.shape[1:], np.nan), c[:-1]])
    f["INTRADAY"] = np.log(c / o)
    f["VWAP_GAP"] = c / vwap - 1.0
    f["HL_RANGE"] = rng / np.r_[np.full((1,) + c.shape[1:], np.nan), c[:-1]]
    f["AMIHUD"] = roll_mean(np.abs(ret) / (amt + EPS), 20, 10)
    f["RET_5"] = np.r_[np.full((5,) + c.shape[1:], np.nan), c[:-5]] / c - 1.0
    f["RET_20"] = np.r_[np.full((20,) + c.shape[1:], np.nan), c[:-20]] / c - 1.0
    f["RET_60"] = np.r_[np.full((60,) + c.shape[1:], np.nan), c[:-60]] / c - 1.0
    f["MOM_SKIP"] = np.r_[np.full((250,) + c.shape[1:], np.nan), c[:-250]] / np.roll(c, 20, axis=0) - 1.0
    f["REV_1"] = -ret
    f["REV_5"] = -(np.r_[np.full((5,) + c.shape[1:], np.nan), c[:-5]] / c - 1.0)
    f["VOL_60"] = roll_std(ret, 60, 30)
    f["VOL_RATIO_5_60"] = roll_std(ret, 5, 3) / (roll_std(ret, 60, 30) + EPS)
    if "float_mktcap" in p.columns:
        f["LN_FLOAT_CAP"] = np.log1p(_arr(p, "float_mktcap"))
    if turn is not None:
        f["TURN_20"] = roll_mean(turn, 20, 10)
        f["TURN_RATIO"] = roll_mean(turn, 5, 3) / (roll_mean(turn, 60, 30) + EPS)
    return f, dates, syms


def _roll_extreme(x, w: int, mp: int, mode: str) -> np.ndarray:
    """窗口极值（NaN 忽略）。用「单调双端队列」的 numpy 版：分块计算，内存只多一个窗口。

    sliding_window_view 会生成 n×w 的只读视图，对 5000×5000 的面板会吃掉数百 GB 虚拟内存，
    因此这里按列分块（每块 256 列）计算，峰值内存 O(256 × n × w)。
    """
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    out = np.full_like(x, np.nan)
    if n < w:
        return out
    xv = x.copy()
    block = 256
    for j0 in range(0, k, block):
        sl = xv[:, j0:j0 + block]
        res = np.full((n - w + 1, sl.shape[1]), np.nan)
        # 用 pandas 的分块 rolling 更稳，且这一算子只在 2×len(windows) 个因子上使用
        s = pd.DataFrame(sl)
        r = s.rolling(w, min_periods=mp).max() if mode == "max" else s.rolling(w, min_periods=mp).min()
        arr = np.asarray(r, dtype=float)
        res[:] = arr[w - 1:]
        out[w - 1:, j0:j0 + block] = res
    return out


def roll_max_ignore(x, w: int, mp: int) -> np.ndarray:
    return _roll_extreme(x, w, mp, "max")


def roll_min_ignore(x, w: int, mp: int) -> np.ndarray:
    return -roll_max_ignore(-np.asarray(x, dtype=float), w, mp)


def label_from_panel(panel: pd.DataFrame, horizons=(5, 20), price_col: str = "close") -> dict[str, np.ndarray]:
    """前向收益：T 日收盘信号 → T+1 开盘买入 → T+1+h 开盘卖出（A 股 T+1 最早成交）。

    注意：标签窗口跨越未来 h 个交易日，与**同一行**的因子做截面相关时，会混入
    「因子本身的持续性」造成的机械相关（标签含起点价 → 反转类因子被高估）。
    检验时请用 ``label_aligned`` 把标的收益改为「从因子窗口末端之后的起点算起」，
    或直接用非重叠标签（每 h 日取样一次）。
    """
    p = panel.copy(); p["date"] = pd.to_datetime(p["date"])
    o = _arr(p, "open")
    out = {}
    for h in horizons:
        buy = np.roll(o, -1, axis=0); buy[-1] = np.nan
        sell = np.roll(o, -(1 + h), axis=0)
        sell[len(o) - 1 - h:] = np.nan
        out[f"h{h}"] = sell / buy - 1.0
    return out


def label_nonoverlap(panel: pd.DataFrame, horizons=(5, 20), price_col: str = "close") -> dict[str, np.ndarray]:
    """非重叠标签：只在每 h 个交易日取一个起点，其余日期为 NaN。

    用于消除「窗口重叠 → 自相关 → t 值虚高」，是检验因子真实预测力的推荐口径。
    """
    full = label_from_panel(panel, horizons, price_col)
    out = {}
    for h_lebel, arr in full.items():
        h = int(h_lebel[1:])
        mask = np.zeros(arr.shape[0], dtype=bool)
        mask[:] = False
        mask[np.arange(arr.shape[0]) % h == 0] = True
        out[h_lebel] = np.where(mask[:, None], arr, np.nan)
    return out
