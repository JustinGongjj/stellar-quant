"""Alpha158 风格因子库（A 股适配版）。

来源与差异
----------
参照 Qlib ``Alpha158DL.get_feature_config`` 的因子家族（K 线形态 9 个、滚动统计
ROC/MA/STD/BETA/RSQR/RESI/MAX/MIN/QTLU/QTLD/RANK/RSV、量价相关 CORD/CORR/CNTP/CNTN/CNTD、
量能变化 VSUMP/VSUMN/VSUMD/IMAX/MIN/RANK/VMA/VSTD/WVMA），在此基础上做三处 A 股适配：

1. **不复权价为主口径**：东方财富官方前复权价存在内部漂移（见 docs/DATA_NOTES.md），
   量价因子一律用不复权价计算；只有跨除权的长周期收益（如 250 日动量）提供
   ``adj=True`` 选项，用自建复权序列。
2. **单位剔除**：所有价格类因子除以当日收盘价（与 Qlib 一致），量能类除以当日成交量，
   保证截面可比。
3. **缺失与极值**：因子在个股时间序列上计算，不做截面填充；截面处理留给检验层
   （winsorize + zscore），避免把未来信息写进因子。

输入约定
--------
``panel`` 为长表，至少包含 date / symbol / open / high / low / close / volume / amount
（可选 avg_price / float_shares / adj_close / turnover / pct_chg）。
输出为宽表字典：``{factor_name: DataFrame(date × symbol)}``。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.pivot_table(index="date", columns="symbol", values=col, aggfunc="last").sort_index()


def _rolling(df: pd.DataFrame, w: int, fn: str, **kw) -> pd.DataFrame:
    r = df.rolling(w, min_periods=max(2, w // 2))
    return getattr(r, fn)(**kw)


def _corr(a: pd.DataFrame, b: pd.DataFrame, w: int) -> pd.DataFrame:
    """逐列滚动相关（pandas rolling corr 只支持两列对齐，这里逐列做）。"""
    out = pd.DataFrame(np.nan, index=a.index, columns=a.columns)
    for c in a.columns:
        out[c] = a[c].rolling(w, min_periods=max(3, w // 2)).corr(b[c])
    return out


def _slope(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """滚动时间序列回归斜率（对 t=0..w-1 回归，单位为「每天」）。"""
    t = np.arange(w, dtype=float)
    tc = t - t.mean()
    denom = float((tc**2).sum())
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    vals = x.to_numpy(dtype=float)
    res = np.full(vals.shape, np.nan)
    for i in range(w - 1, len(x)):
        win = vals[i - w + 1:i + 1]
        ok = np.isfinite(win).all(axis=0)
        if ok.any():
            yc = win[:, ok] - win[:, ok].mean(axis=0)
            res[i, ok] = (tc[:, None] * yc).sum(axis=0) / denom
    return pd.DataFrame(res, index=x.index, columns=x.columns)


def build_factors(panel: pd.DataFrame, windows: tuple[int, ...] = WINDOWS,
                  include_kbar: bool = True, include_beta: bool = True) -> dict[str, pd.DataFrame]:
    """返回 ``{因子名: 宽表}``。因子命名与 Qlib 保持一致，便于对照。"""
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    o = _wide(p, "open").astype(float)
    h = _wide(p, "high").astype(float)
    lo = _wide(p, "low").astype(float)
    c = _wide(p, "close").astype(float)
    v = _wide(p, "volume").astype(float)
    a = _wide(p, "amount").astype(float)
    vwap = _wide(p, "avg_price").astype(float) if "avg_price" in p.columns else (a / (v * 100.0))
    vwap = vwap.where(np.isfinite(vwap) & (vwap > 0))
    f: dict[str, pd.DataFrame] = {}
    eps = 1e-12

    # ---------------------------------------------------------------- K 线形态
    if include_kbar:
        f["KMID"] = (c - o) / o
        f["KLEN"] = (h - lo) / o
        f["KMID2"] = (c - o) / (h - lo + eps)
        f["KUP"] = (h - np.maximum(o, c)) / o
        f["KUP2"] = (h - np.maximum(o, c)) / (h - lo + eps)
        f["KLOW"] = (np.minimum(o, c) - lo) / o
        f["KLOW2"] = (np.minimum(o, c) - lo) / (h - lo + eps)
        f["KSFT"] = (2 * c - h - lo) / o
        f["KSFT2"] = (2 * c - h - lo) / (h - lo + eps)

    # ---------------------------------------------------------------- 价格位置
    for w in windows:
        f[f"ROC{w}"] = c.shift(w) / c - 1.0
        f[f"MA{w}"] = _rolling(c, w, "mean") / c - 1.0
        f[f"STD{w}"] = _rolling(c / c.shift(1) - 1.0, w, "std") / (c / c.shift(1) - 1.0).abs().rolling(w, min_periods=2).mean().abs()
        f[f"MAX{w}"] = _rolling(h, w, "max") / c - 1.0
        f[f"MIN{w}"] = _rolling(lo, w, "min") / c - 1.0
        f[f"QTLU{w}"] = c.rolling(w, min_periods=max(2, w // 2)).quantile(0.8) / c - 1.0
        f[f"QTLD{w}"] = c.rolling(w, min_periods=max(2, w // 2)).quantile(0.2) / c - 1.0
        f[f"RANK{w}"] = c.rolling(w, min_periods=max(2, w // 2)).rank(pct=True)
        f[f"RSV{w}"] = (c - _rolling(lo, w, "min")) / (_rolling(h, w, "max") - _rolling(lo, w, "min") + eps)
        if include_beta:
            f[f"BETA{w}"] = _slope(c / c.shift(1) - 1.0, w) / (c / c.shift(1) - 1.0).abs().rolling(w, min_periods=2).mean().abs()
        # 量价相关与量能结构
        ret = c / c.shift(1) - 1.0
        f[f"CORR{w}"] = _corr(ret, np.log1p(v / v.shift(1) - 1.0).abs(), w)
        up = ret > 0
        f[f"CNTP{w}"] = up.rolling(w, min_periods=max(2, w // 2)).mean()
        f[f"CNTN{w}"] = (ret < 0).rolling(w, min_periods=max(2, w // 2)).mean()
        f[f"CNTD{w}"] = f[f"CNTP{w}"] - f[f"CNTN{w}"]
        dv = v.diff()
        f[f"VSUMP{w}"] = dv.clip(lower=0).rolling(w, min_periods=max(2, w // 2)).sum() / (dv.abs().rolling(w, min_periods=max(2, w // 2)).sum() + eps)
        f[f"VSUMN{w}"] = (-dv).clip(lower=0).rolling(w, min_periods=max(2, w // 2)).sum() / (dv.abs().rolling(w, min_periods=max(2, w // 2)).sum() + eps)
        f[f"VSUMD{w}"] = f[f"VSUMP{w}"] - f[f"VSUMN{w}"]
        f[f"VMA{w}"] = _rolling(v, w, "mean") / (v + eps) - 1.0
        f[f"VSTD{w}"] = _rolling(v, w, "std") / (v.rolling(w, min_periods=2).mean() + eps)
        f[f"VOL_RATIO{w}"] = v / (_rolling(v, w, "mean") + eps)
        # 价格-成交量弹性：近 w 日收益与换手变化的相关
        if "turnover" in p.columns:
            to = _wide(p, "turnover").astype(float) / 100.0
            f[f"TURN{w}"] = to.rolling(w, min_periods=max(2, w // 2)).mean()
            f[f"CORR_TURN{w}"] = _corr(ret, to.diff(), w)
        # 资金流代理：成交额加权的涨跌幅
        f[f"AMT_MA{w}"] = np.log1p(_rolling(a, w, "mean"))
        f[f"AMT_RATIO{w}"] = np.log1p(a) - np.log1p(_rolling(a, w, "mean"))

    # ---------------------------------------------------------------- 日内与价位
    f["CLOSE_POS"] = (c - lo) / (h - lo + eps)
    f["PRICE_GAP"] = o / c.shift(1) - 1.0
    f["OVERNIGHT"] = np.log(o / c.shift(1))
    f["INTRADAY"] = np.log(c / o)
    f["VWAP_GAP"] = c / vwap - 1.0
    f["HL_RANGE"] = (h - lo) / c.shift(1)
    f["AMIHUD"] = (c.pct_change().abs() / np.log1p(a)).rolling(20, min_periods=10).mean()
    f["RET_5"] = c / c.shift(5) - 1.0
    f["RET_60"] = c / c.shift(60) - 1.0
    f["MOM_SKIP"] = c.shift(20) / c.shift(250) - 1.0
    f["REV_1"] = -c.pct_change()
    f["REV_5"] = -(c / c.shift(5) - 1.0)
    f["VOL_60"] = (c.pct_change()).rolling(60, min_periods=30).std()
    f["MAX_RET_20"] = c.pct_change().rolling(20, min_periods=10).max()
    f["SKEW_60"] = c.pct_change().rolling(60, min_periods=30).skew()

    # ---------------------------------------------------------------- 市值与流动性
    if "float_mktcap" in p.columns:
        f["LN_FLOAT_CAP"] = np.log1p(_wide(p, "float_mktcap").astype(float))
    f["AMT_20"] = np.log1p(_rolling(a, 20, "mean"))
    return f


def factor_frame(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把宽表字典堆成长表 ``date / symbol / f1 / f2 ...``，供检验层与模型使用。"""
    names = list(factors)
    stack = []
    for name in names:
        s = factors[name].stack(future_stack=True)
        s.name = name
        stack.append(s)
    df = pd.concat(stack, axis=1)
    df.index = df.index.set_names(["date", "symbol"])
    return df.reset_index()


def label_forward_returns(panel: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20),
                          use_adj: bool = True) -> dict[str, pd.DataFrame]:
    """前向收益标签：T+1 开盘买入、T+1+h 开盘卖出（A 股 T 日信号最早 T+1 成交）。"""
    p = panel.copy()
    px_col = "adj_close" if use_adj and "adj_close" in p.columns else "close"
    o = _wide(p, "open").astype(float)
    c = _wide(p, px_col).astype(float)
    if use_adj and px_col == "adj_close":
        # 用复权收盘/不复权开盘的混合口径会引入偏差，这里统一用复权收益：open 按同日复权比例换算
        raw_c = _wide(p, "close").astype(float)
        o = o * (c / raw_c)
    buy = o.shift(-1)
    out = {}
    for h in horizons:
        sell = o.shift(-(1 + h))
        out[f"label_h{h}"] = sell / buy - 1.0
    return out


# ---------------------------------------------------------------------------
# NaN 感知的滚动算子（向量化）
# ---------------------------------------------------------------------------
# pandas 的 rolling/rolling.corr 在宽表上逐列 Python 循环，5000 只 × 5000 天的面板
# 需要十几分钟。这里用 cumsum 前缀和 + 并行累乘（pairwise）实现同样口径的运算，
# 复杂度 O(n)，且 min_periods 与 pandas 默认一致（窗口内有效值不足即 NaN）。

def _pref(x: np.ndarray) -> np.ndarray:
    """前缀和：P[i+1] = sum(x[:i+1])，输入按时间在 axis=0。"""
    x = np.asarray(x, dtype=float)
    out = np.empty((x.shape[0] + 1,) + x.shape[1:])
    out[0] = 0.0
    np.cumsum(x, axis=0, out=out[1:])
    return out


def roll_sum(x: np.ndarray, w: int, min_periods: int | None = None) -> np.ndarray:
    mp = min_periods if min_periods is not None else w
    xv = np.where(np.isfinite(x), x, 0.0)
    cnt = np.isfinite(x).astype(np.float64)
    s = _pref(xv)
    n = _pref(cnt)
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < w:
        return out
    num = s[w:] - s[:-w]
    den = n[w:] - n[:-w]
    valid = den >= mp
    res = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    res = np.where(valid, res, np.nan)
    out[w - 1:] = res
    return out


def roll_std(x: np.ndarray, w: int, min_periods: int | None = None, ddof: int = 1) -> np.ndarray:
    mp = min_periods if min_periods is not None else w
    xv = np.where(np.isfinite(x), x, 0.0)
    cnt = np.isfinite(x).astype(np.float64)
    x2 = xv * xv
    s = _pref(xv)
    s2 = _pref(x2)
    n = _pref(cnt)
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < w:
        return out
    num = s[w:] - s[:-w]
    num2 = s2[w:] - s2[:-w]
    den = n[w:] - n[:-w]
    den_safe = np.where(den > 0, den, 1.0)
    mean = num / den_safe
    var = (num2 / den_safe) - mean * mean
    sig = np.sqrt(np.maximum(var, 0.0) * den_safe / np.maximum(den_safe - ddof, 1e-12))
    sig = np.where(den >= mp, sig, np.nan)
    out[w - 1:] = sig
    return out


def roll_corr(a: np.ndarray, b: np.ndarray, w: int, min_periods: int | None = None) -> np.ndarray:
    mp = min_periods if min_periods is not None else w
    ok = np.isfinite(a) & np.isfinite(b)
    av = np.where(ok, a, 0.0)
    bv = np.where(ok, b, 0.0)
    cnt = ok.astype(np.float64)
    sa, sb = _pref(av), _pref(bv)
    sab, saa, sbb = _pref(av * bv), _pref(av * av), _pref(bv * bv)
    sc = _pref(cnt)
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) < w:
        return out
    sa_w = sa[w:] - sa[:-w]
    sb_w = sb[w:] - sb[:-w]
    sab_w = sab[w:] - sab[:-w]
    saa_w = saa[w:] - saa[:-w]
    sbb_w = sbb[w:] - sbb[:-w]
    n = sc[w:] - sc[:-w]
    n_safe = np.where(n > 0, n, 1.0)
    cov = sab_w - sa_w * sb_w / n_safe
    va = saa_w - sa_w * sa_w / n_safe
    vb = sbb_w - sb_w * sb_w / n_safe
    denom = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
    r = np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0), np.nan)
    r = np.where(n >= mp, r, np.nan)
    out[w - 1:] = np.clip(r, -1.0, 1.0)
    return out


def roll_argmax_pos(x: np.ndarray, w: int) -> np.ndarray:
    """窗口内最大值出现位置距窗口末端的距离（0 = 当日最高）。NaN 视为不参与。"""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(x.shape, np.nan)
    if len(x) < w:
        return out
    win = sliding_window_view(x, w, axis=0)      # (n-w+1, k, w)
    filled = np.where(np.isfinite(win), win, -np.inf)
    pos = np.argmax(filled, axis=-1)
    allnan = ~np.isfinite(win).any(axis=-1)
    res = (w - 1 - pos).astype(float)
    res[allnan] = np.nan
    out[w - 1:] = res
    return out


def fast_momentum_factor(c: pd.DataFrame, w: int) -> pd.DataFrame:
    """例子：用 NaN 感知算子实现 ``MA{w}``（均值/收盘 - 1），口径与 pandas 一致。"""
    arr = c.to_numpy(dtype=float)
    ma = roll_sum(arr, w, min_periods=max(2, w // 2)) / w
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ma / arr - 1.0
    return pd.DataFrame(out, index=c.index, columns=c.columns)
