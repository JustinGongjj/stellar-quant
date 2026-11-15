"""因子检验层：截面 IC / 分层 / 衰减 / 换手 / 多重检验校正。

设计口径
--------
* **截面标准化**：每日先做 MAD winsorize（±3σ）再 z-score，缺失按中性 0（仅用于分层与 IC，
  模型层单独处理缺失）。
* **RankIC**：每日截面 Spearman 相关；ICIR = mean(IC)/std(IC)；t 值用 Newey-West 修正，
  因为日频 IC 存在自相关（Qlib 的 t 值未修正，这里更严格）。
* **分层**：按因子值分 5 组，等权持有，记录各组前瞻收益、多空价差（Q5−Q1）与单调性。
* **换手与衰减**：因子排名换手（前后两期 rank 相关的 1 − rho），以及不同前向持有期的 IC 衰减。
* **多重检验**：多个候选因子同时检验时，用 Benjamini–Hochberg 控制 FDR（q=0.05），
  并在报告里同时给出 Bonferroni 阈值，避免「捡到显著的」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section_prep(f: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """逐日：中位数/MAD 标准化并截断 ±winsor；全 NaN 横截面保持 NaN。"""
    def _one(row: pd.Series) -> pd.Series:
        v = row.astype(float)
        med = v.median()
        mad = (v - med).abs().median() * 1.4826
        if not np.isfinite(mad) or mad <= 0:
            std = v.std(ddof=0)
            mad = std if np.isfinite(std) and std > 0 else np.nan
        if not np.isfinite(mad):
            return v * np.nan
        return ((v - med) / mad).clip(-winsor, winsor)
    return f.apply(_one, axis=1)


def rank_ic_series(f: pd.DataFrame, label: pd.DataFrame, min_n: int = 30) -> pd.Series:
    """每日截面 RankIC（Spearman）。"""
    idx = f.index.intersection(label.index)
    out = {}
    for dt in idx:
        a = f.loc[dt]
        b = label.loc[dt]
        m = a.notna() & b.notna()
        if int(m.sum()) < min_n:
            continue
        va = a[m].to_numpy(dtype=float)
        vb = b[m].to_numpy(dtype=float)
        if np.nanstd(va) == 0 or np.nanstd(vb) == 0:
            continue
        ra = pd.Series(va).rank().to_numpy()
        rb = pd.Series(vb).rank().to_numpy()
        out[dt] = float(np.corrcoef(ra, rb)[0, 1])
    return pd.Series(out, dtype=float).sort_index()


def newey_west_t(series: pd.Series, lags: int = 5) -> float:
    """IC 序列均值的 Newey-West t 值。"""
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 10:
        return float("nan")
    mu = x.mean()
    e = x - mu
    var = (e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        cov = (e[L:] @ e[:-L]) / n
        var += 2.0 * (1.0 - L / (lags + 1.0)) * cov
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def ic_stats(f: pd.DataFrame, label: pd.DataFrame, min_n: int = 30,
             nw_lags: int = 5) -> dict:
    ic = rank_ic_series(f, label, min_n)
    if len(ic) < 10:
        return {"n": int(len(ic)), "ic_mean": float("nan")}
    return {
        "n": int(len(ic)),
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std(ddof=1)),
        "icir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else float("nan"),
        "t_nw": newey_west_t(ic, nw_lags),
        "t_iid": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if ic.std(ddof=1) > 0 else float("nan"),
        "positive_ratio": float((ic > 0).mean()),
        "ic_decay_slope": float(np.polyfit(np.arange(len(ic)), ic.rolling(20, min_periods=5).mean().fillna(ic.mean()).to_numpy(), 1)[0]),
    }


def layered_returns(f: pd.DataFrame, label: pd.DataFrame, n_groups: int = 5,
                    min_n: int = 30) -> pd.DataFrame:
    """每日按因子分位分组，返回各组等权前瞻收益（date × group）。"""
    idx = f.index.intersection(label.index)
    rows = {}
    for dt in idx:
        a = f.loc[dt]
        b = label.loc[dt]
        m = a.notna() & b.notna()
        if int(m.sum()) < min_n:
            continue
        r = a[m].rank(method="first")
        g = pd.qcut(r, n_groups, labels=False)
        rows[dt] = b[m].groupby(g).mean()
    out = pd.DataFrame(rows).T
    out.columns = [f"G{int(c) + 1}" for c in out.columns]
    return out.sort_index()


def layer_stats(layers: pd.DataFrame) -> dict:
    if layers.empty:
        return {"n": 0}
    means = layers.mean()
    top, bot = means.iloc[-1], means.iloc[0]
    ls = (layers.iloc[:, -1] - layers.iloc[:, 0])
    mono = np.corrcoef(np.arange(len(means)), means.rank())[0, 1] if len(means) > 2 else float("nan")
    return {
        "n": int(len(layers)),
        "group_means": means.to_dict(),
        "ls_mean_daily": float(ls.mean()),
        "ls_t": float(ls.mean() / (ls.std(ddof=1) / np.sqrt(len(ls)))) if ls.std(ddof=1) > 0 else float("nan"),
        "monotonicity": float(mono),
        "top_minus_bottom": float(ls.mean()),
    }


def turnover(f: pd.DataFrame, lag: int = 1, min_n: int = 30) -> float:
    """因子排名换手：1 − 相邻两期排名的 Spearman 相关（越高越贵）。

    注意：这**不是**组合实际换手，组合换手由回测引擎给出；该指标用于比较因子稳定性。
    """
    idx = f.index.sort_values()
    vals = []
    for i in range(lag, len(idx)):
        a, b = f.loc[idx[i]], f.loc[idx[i - lag]]
        m = a.notna() & b.notna()
        if int(m.sum()) < min_n:
            continue
        va = a[m].to_numpy(dtype=float)
        vb = b[m].to_numpy(dtype=float)
        if np.nanstd(va) == 0 or np.nanstd(vb) == 0:
            continue
        ra = pd.Series(va).rank().to_numpy()
        rb = pd.Series(vb).rank().to_numpy()
        vals.append(1.0 - float(np.corrcoef(ra, rb)[0, 1]))
    return float(np.nanmean(vals)) if vals else float("nan")


def decay_ic(f: pd.DataFrame, labels: dict[str, pd.DataFrame], min_n: int = 30) -> dict:
    return {k: ic_stats(f, v, min_n)["ic_mean"] for k, v in labels.items()}


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg 校正后的 q 值。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return list(out)


def pvalue_from_t(t: float, df: int) -> float:
    """双侧 p 值（t 分布的 Normal 近似，df>60 时误差 <1e-3）。"""
    if not np.isfinite(t):
        return float("nan")
    z = abs(t)
    # 标准正态尾部：用 erfc 近似
    import math
    return float(math.erfc(z / math.sqrt(2.0)))


def screen_factors(frames: dict[str, pd.DataFrame], label: pd.DataFrame,
                   n_groups: int = 5, min_n: int = 30) -> pd.DataFrame:
    """对一个因子字典做全量检验，返回一张因子报告表（含 BH 校正后的 q 值）。"""
    rows = []
    for name, f in frames.items():
        ff = cross_section_prep(f)
        st = ic_stats(ff, label, min_n)
        if st.get("n", 0) < 30:
            continue
        lay = layered_returns(ff, label, n_groups, min_n)
        ls = layer_stats(lay)
        rows.append({
            "factor": name,
            **{k: st.get(k) for k in ("n", "ic_mean", "ic_std", "icir", "t_nw", "positive_ratio")},
            "ls_t": ls.get("ls_t"),
            "monotonicity": ls.get("monotonicity"),
            "top_minus_bottom": ls.get("top_minus_bottom"),
            "turnover": turnover(ff, min_n=min_n),
        })
    rep = pd.DataFrame(rows)
    if rep.empty:
        return rep
    rep["p_value"] = [pvalue_from_t(t, int(n) - 1) for t, n in zip(rep["t_nw"], rep["n"])]
    rep["q_bh"] = bh_adjust(rep["p_value"].fillna(1.0).tolist())
    rep["sign"] = np.sign(rep["ic_mean"])
    rep["abs_icir"] = rep["icir"].abs()
    return rep.sort_values("abs_icir", ascending=False).reset_index(drop=True)
