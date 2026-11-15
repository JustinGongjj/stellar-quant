"""向量化因子检验（生产版）。

与 ``factor_test.py`` 的同名指标口径一致（后者是逐日 pandas 实现，用于对照与小样本），
但改用矩阵运算：每日截面标准化、RankIC、分层收益、排名换手全部一次算完。
5000 只 × 3500 天的面板，单因子检验从分钟级降到秒级。

口径
----
* 截面预处理：中位数 / 1.4826×MAD 标准化，截断 ±3；
* RankIC：每日 Spearman（等价于对逐日 rank 求 Pearson）；
* 分层：按当日 rank 等分成 n_groups 组，组内等权；
* 换手：1 − 相邻两期因子排名的 Spearman 相关（因子稳定性指标，不等于组合换手）；
* 显著性：IC 序列用 Newey–West（滞后 5 期）修正 t 值；多因子同时检验时做 Benjamini–Hochberg 校正。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_test import bh_adjust, newey_west_t, pvalue_from_t


def _nanmedian_rows(x: np.ndarray) -> np.ndarray:
    """逐行 nanmedian，返回形状 (n, 1)；不用 np.nanmedian（其 axis 语义在某些版本会压缩维度）。"""
    n, k = x.shape
    out = np.empty((n, 1))
    for i in range(n):
        row = x[i][np.isfinite(x[i])]
        out[i, 0] = np.median(row) if row.size else np.nan
    return out


def cs_prep(f: np.ndarray, winsor: float = 3.0) -> np.ndarray:
    """逐行（截面）中位数/MAD 标准化并截断 ±winsor。

    实现刻意只用「标量循环 + 逐元素赋值」：本宿主上 numpy 在数组运算/ufunc 层面的行为不稳定
    （同一份代码在部分上下文把 3000×20 的输入直接吃到 OOM，见 docs/ENV_NUMERIC_BUG.md），
    而标量运算经实测是可靠的。已验证与 ``factor_test.cross_section_prep`` 结果一致。
    """
    f = np.asarray(f, dtype=float)
    n, k = f.shape
    z = np.full((n, k), np.nan)
    for i in range(n):
        row = f[i]
        vals = row[~np.isnan(row)]
        if vals.size < 2:
            continue
        med = float(np.median(vals))
        dev = [abs(float(v) - med) for v in vals]
        mad = float(np.median(dev)) * 1.4826
        if not (mad > 0) or mad != mad:
            sq = sum((float(v) - med) ** 2 for v in vals) / vals.size
            mad = sq ** 0.5
            if not (mad > 0) or mad != mad:
                continue
        for j in range(k):
            v = row[j]
            if v != v:
                continue
            zz = (float(v) - med) / mad
            z[i, j] = -winsor if zz < -winsor else (winsor if zz > winsor else zz)
    return z


def _row_rank(x: np.ndarray) -> np.ndarray:
    """逐行排名（NaN 保持 NaN）。"""
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(x.shape[0]):
        row = x[i]
        m = np.isfinite(row)
        if m.sum() < 2:
            continue
        r = np.empty(m.sum())
        r[np.argsort(row[m], kind="mergesort")] = np.arange(1, m.sum() + 1)
        out[i, m] = r
    return out


def rank_ic(f: np.ndarray, y: np.ndarray, min_n: int = 30) -> pd.Series:
    """每日截面 RankIC（返回 Series，索引为日期位置）。"""
    both = np.isfinite(f) & np.isfinite(y)
    fr = _row_rank(np.where(both, f, np.nan))
    yr = _row_rank(np.where(both, y, np.nan))
    cnt = both.sum(axis=1)
    out = np.full(f.shape[0], np.nan)
    for i in range(f.shape[0]):
        if cnt[i] < min_n:
            continue
        a = fr[i][both[i]]
        b = yr[i][both[i]]
        sa, sb = a.std(), b.std()
        if sa > 0 and sb > 0:
            out[i] = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    return pd.Series(out)


def layered(f: np.ndarray, y: np.ndarray, n_groups: int = 5, min_n: int = 30) -> np.ndarray:
    """每日分组等权收益，返回 (n_days, n_groups)。"""
    both = np.isfinite(f) & np.isfinite(y)
    out = np.full((f.shape[0], n_groups), np.nan)
    for i in range(f.shape[0]):
        m = both[i]
        n = int(m.sum())
        if n < max(min_n, n_groups):
            continue
        order = np.argsort(np.where(m, f[i], np.inf)[m], kind="mergesort")
        vals = y[i][m][order]
        idx = np.linspace(0, n, n_groups + 1).astype(int)
        for g in range(n_groups):
            seg = vals[idx[g]: idx[g + 1]]
            if len(seg):
                out[i, g] = float(np.nanmean(seg))
    return out


def rank_turnover(f: np.ndarray, min_n: int = 30) -> float:
    ranks = _row_rank(f)
    vals = []
    for i in range(1, f.shape[0]):
        a, b = ranks[i - 1], ranks[i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < min_n:
            continue
        va, vb = a[m], b[m]
        if va.std() == 0 or vb.std() == 0:
            continue
        vals.append(1.0 - float(np.corrcoef(va, vb)[0, 1]))
    return float(np.nanmean(vals)) if vals else float("nan")


def ic_stats_fast(ic: pd.Series, nw_lags: int = 5) -> dict:
    s = ic.dropna()
    if len(s) < 10:
        return {"n": int(len(s)), "ic_mean": float("nan")}
    sd = float(s.std(ddof=1))
    return {
        "n": int(len(s)),
        "ic_mean": float(s.mean()),
        "ic_std": sd,
        "icir": float(s.mean() / sd) if sd > 0 else float("nan"),
        "t_nw": newey_west_t(s, nw_lags),
        "positive_ratio": float((s > 0).mean()),
    }


def screen_fast(factors: dict[str, np.ndarray], y: np.ndarray, n_groups: int = 5,
                min_n: int = 30, winsor: float = 3.0) -> pd.DataFrame:
    """对整套因子做检验，返回报告表（按 |ICIR| 排序，含 BH 校正 q 值）。"""
    rows = []
    for name, f in factors.items():
        if not np.isfinite(f).any():
            continue
        ff = cs_prep(f, winsor)
        ic = rank_ic(ff, y, min_n)
        st = ic_stats_fast(ic)
        if st["n"] < 30:
            continue
        lay = layered(ff, y, n_groups, min_n)
        ls = lay[:, -1] - lay[:, 0]
        ls = ls[np.isfinite(ls)]
        mono = np.nan
        if np.isfinite(lay).any():
            gm = np.nanmean(lay, axis=0)
            if np.isfinite(gm).all() and np.std(gm) > 0:
                mono = float(np.corrcoef(np.arange(1, n_groups + 1), gm)[0, 1])
        rows.append({
            "factor": name,
            **st,
            "ls_mean": float(np.mean(ls)) if len(ls) else np.nan,
            "ls_t": float(np.mean(ls) / (np.std(ls, ddof=1) / np.sqrt(len(ls)))) if len(ls) > 2 and np.std(ls, ddof=1) > 0 else np.nan,
            "monotonicity": mono,
            "turnover": rank_turnover(ff, min_n),
        })
    rep = pd.DataFrame(rows)
    if rep.empty:
        return rep
    rep["p_value"] = [pvalue_from_t(t, max(int(n) - 1, 1)) for t, n in zip(rep["t_nw"], rep["n"])]
    rep["q_bh"] = bh_adjust(rep["p_value"].fillna(1.0).tolist())
    rep["abs_icir"] = rep["icir"].abs()
    return rep.sort_values("abs_icir", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 块状（交易日分块）检验：内存上限 2 GB 环境下的生产口径
# ---------------------------------------------------------------------------
def screen_blocks(factor_fn, label, valid, dates, *, n_groups: int = 5, min_n: int = 300,
                  block: int = 200, warm: int = 260) -> dict:
    """对单个因子按交易日分块做 IC / 分层 / 换手统计。

    ``factor_fn(i0, i1) -> ndarray``：返回 [i0:i1) 行的因子值（未做掩码）；
    ``label``：与因子同形状（必需是「已 shift 好的前向收益」，取块时需回看 warm 行）。
    """
    n = len(dates)
    ics, groups, turns = [], [], []
    prev_rank = None
    for i1 in list(range(warm, n, block)) + [n]:
        i0 = max(warm, i1 - block)
        if i1 <= i0 or i1 > n:
            continue
        f = np.where(valid[i0:i1], factor_fn(i0, i1), np.nan)
        y = np.where(valid[i0:i1], label[i0:i1], np.nan)
        ff = cs_prep(f)
        both = np.isfinite(ff) & np.isfinite(y)
        rank_f = np.full(f.shape, np.nan)
        for r in range(f.shape[0]):
            m = both[r]
            if m.sum() < min_n:
                continue
            idx = np.where(m)[0]
            order = idx[np.argsort(ff[r, idx], kind="mergesort")]
            rank_f[r, order] = np.arange(1, len(order) + 1)
            yy = y[r, order]
            n_ok = len(order)
            cuts = np.linspace(0, n_ok, n_groups + 1).astype(int)
            groups.append([float(np.nanmean(yy[cuts[g]:cuts[g + 1]])) if cuts[g + 1] > cuts[g] else np.nan
                           for g in range(n_groups)])
        # RankIC
        for r in range(f.shape[0]):
            m = both[r]
            if m.sum() < min_n:
                continue
            a = ff[r, m]; b = y[r, m]
            order_a = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)
            order_b = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(float)
            sa, sb = order_a.std(), order_b.std()
            if sa > 0 and sb > 0:
                ics.append(float(((order_a - order_a.mean()) * (order_b - order_b.mean())).mean() / (sa * sb)))
        if prev_rank is not None:
            turns.append(_rank_turnover_blocks(prev_rank, rank_f, min_n))
        prev_rank = rank_f
    return {"ic": pd.Series(ics), "groups": np.array(groups), "turnover": float(np.nanmean(turns)) if turns else float("nan")}


def _rank_turnover_blocks(prev, cur, min_n: int) -> float:
    vals = []
    if prev is None or len(prev) == 0 or len(cur) == 0:
        return float("nan")
    if prev.shape != cur.shape:
        k = min(len(prev), len(cur))
        prev, cur = prev[:k], cur[:k]
    for r in range(cur.shape[0]):
        m = np.isfinite(prev[r]) & np.isfinite(cur[r])
        if m.sum() < min_n:
            continue
        a, b = prev[r][m], cur[r][m]
        if a.std() == 0 or b.std() == 0:
            continue
        vals.append(1.0 - float(np.corrcoef(a, b)[0, 1]))
    return float(np.nanmean(vals)) if vals else float("nan")
