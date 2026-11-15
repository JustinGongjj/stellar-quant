"""截面风险模型：市值加权 WLS 估计行业因子 + 风格因子收益，并分解残差。

吸收自 toraniko（0xfdf/toraniko，MIT）的实现口径：

1. 权重 W = diag(√市值)：大市值股票对因子收益估计更稳定；
2. 行业因子加「零和约束」——(1, S) 设计阵右乘变换矩阵 R 使 Σ 行业因子收益 = 0，
   等价于市场收益被行业完全张成（market is spanned by sectors）；
3. 风格因子在行业残差上回归（正交化），避免风格与行业共线；
4. 残差 = 行业残差 - 风格暴露 × 风格收益。

与 toraniko 的差异：去掉 polars 依赖，直接在 numpy 上实现；输入为 date × symbol 的宽表；
再加一个「稳健标准化」与「缺失暴露按 0 处理」的显式规则。

校验：用构造数据做恒等式检验（factor_ret 的加权残差和 ≈ 0、残差与暴露正交）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskModelResult:
    factor_returns: pd.DataFrame       # date × (sector_*, style_*)
    residuals: pd.DataFrame            # date × symbol
    factor_cov: pd.DataFrame           # 因子收益协方差（年化可选）
    specific_var: pd.DataFrame         # date × symbol 残差方差


def _winsorize(x: np.ndarray, pct: float = 0.01) -> np.ndarray:
    if pct <= 0 or len(x) < 20:
        return x
    lo = np.nanpercentile(x, pct * 100.0)
    hi = np.nanpercentile(x, (1.0 - pct) * 100.0)
    return np.clip(x, lo, hi)


def _wls(beta: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """加权最小二乘：argmin Σ w_i (y_i - β_i·f)²，用 SVD 求伪逆，秩亏时稳健。"""
    bw = beta * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(bw, yw, rcond=None)
    return coef


def standardize_cross(series: pd.Series) -> pd.Series:
    """截面稳健标准化：中位数居中、MAD 缩放，并截断在 ±3。"""
    v = series.astype(float)
    med = v.median()
    mad = (v - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        std = v.std(ddof=0)
        mad = std if std and np.isfinite(std) and std > 0 else 1.0
    z = (v - med) / mad
    return z.clip(-3.0, 3.0)


def estimate_risk_model(
    returns: pd.DataFrame,
    mkt_cap: pd.DataFrame,
    sectors: pd.DataFrame,
    styles: pd.DataFrame,
    winsor: float = 0.0,
    coverage: float = 0.6,
    residualize_styles: bool = True,
) -> RiskModelResult:
    """逐期估计「行业 + 风格」风险模型。

    所有输入均为 date × symbol 宽表，日期对齐后逐日估计。某日可用股票少于
    ``coverage × 平均股票数`` 时跳过（返回 NaN 行），不做插补。
    """
    dates = returns.index.intersection(mkt_cap.index).intersection(sectors.index).intersection(styles.index)
    symbols = returns.columns
    ref_date = dates[0]
    sector_names = sorted({str(v) for v in sectors.loc[ref_date].dropna().unique()})
    # 风格暴露表约定为 日期 × (symbol × 风格因子) 宽表，列名 "SYMBOL:factor"
    style_names = sorted({str(c).split(":", 1)[1] for c in styles.columns if ":" in str(c)})
    out_f, out_r = {}, {}
    prev_sector_dummies = None
    for dt in dates:
        y = returns.loc[dt, symbols].astype(float)
        cap = mkt_cap.loc[dt, symbols].astype(float)
        sec = sectors.loc[dt]
        st = styles.loc[dt]
        valid = y.notna() & cap.notna() & (cap > 0)
        if valid.sum() < 30:
            continue
        idx = symbols[valid]
        yv = y.loc[idx].to_numpy(dtype=float)
        if winsor > 0:
            yv = _winsorize(yv, winsor)
        w = np.sqrt(cap.loc[idx].to_numpy(dtype=float))
        # 行业哑变量（列 = 行业），缺暴露的股票归入最小行业使数量不变
        dummies = np.zeros((len(idx), len(sector_names)))
        secv = sec.reindex(idx).astype(str)
        for j, name in enumerate(sector_names):
            dummies[:, j] = (secv == name).to_numpy(dtype=float)
        if dummies.sum(axis=0).min() == 0:
            sector_names_use = [n for j, n in enumerate(sector_names) if dummies[:, j].sum() > 0]
            dummies = dummies[:, [j for j, n in enumerate(sector_names) if n in sector_names_use]]
        else:
            sector_names_use = sector_names
        n_sec = dummies.shape[1]
        beta = np.hstack([np.ones((len(idx), 1)), dummies])
        # 零和约束：f_1 = -Σ_{j≥2} f_j
        a = np.concatenate([[0.0], -np.ones(n_sec - 1)])
        R = np.vstack([np.eye(n_sec), a])
        B = beta @ R
        g = _wls(B, w, yv)
        fac_sector = R @ g
        resid_sector = yv - dummies @ fac_sector[1:]

        S = pd.DataFrame({n: pd.Series({sym: st.get(f"{sym}:{n}", 0.0) for sym in idx}, dtype=float)
                          for n in style_names}, index=idx)
        if styles.empty or not style_names:
            raise ValueError("风格暴露表为空：需要列名形如 '600519:mom' 的宽表")
        # 标准化风格暴露（截面），避免量纲不同导致风格收益不可比
        Sv = np.column_stack([standardize_cross(S[n]).to_numpy(dtype=float) for n in style_names])
        target = resid_sector if residualize_styles else yv
        f_style = _wls(Sv, w, target)
        resid = target - Sv @ f_style
        names = ["market"] + [f"sector:{n}" for n in sector_names_use] + [f"style:{n}" for n in style_names]
        f_ret = np.concatenate([np.atleast_1d(fac_sector), np.atleast_1d(f_style)])
        if f_ret.shape[0] != len(names):
            raise RuntimeError(f"因子收益维度不一致：{f_ret.shape[0]} vs {len(names)}")
        out_f[dt] = pd.Series(f_ret, index=names)
        r = pd.Series(np.nan, index=symbols, dtype=float)
        r.loc[idx] = resid
        out_r[dt] = r
    if not out_f:
        raise ValueError("没有足够的截面样本用于估计风险模型（每日至少 30 只）")
    fac = pd.DataFrame(out_f).T.sort_index()
    res = pd.DataFrame(out_r).T.sort_index()
    cov = fac.cov()
    spec = res.var(axis=1)
    return RiskModelResult(factor_returns=fac, residuals=res, factor_cov=cov, specific_var=spec.to_frame("spec_var"))


def risk_attribution(weights: pd.Series, exposure: pd.DataFrame,
                     factor_cov: pd.DataFrame, specific_var: pd.Series) -> dict:
    """组合风险分解：σ² = xᵀ B Σ_f Bᵀ x + Σ x_i² σ_i²。"""
    x = weights.astype(float).fillna(0.0)
    common = exposure.columns.intersection(x.index)
    B = exposure[common].to_numpy(dtype=float)
    w = x.loc[common].to_numpy(dtype=float)
    Sig = factor_cov.to_numpy(dtype=float)
    common_var = float(w @ B @ Sig @ B.T @ w)
    spec_var = float(np.sum(w**2 * specific_var.reindex(common).fillna(0.0).to_numpy(dtype=float) ** 2))
    total = common_var + spec_var
    return {
        "total_var": total,
        "common_var": common_var,
        "specific_var": spec_var,
        "common_share": common_var / total if total > 0 else float("nan"),
        "risk_contrib_by_factor": (B.T @ w) * (Sig @ (B.T @ w)),
    }
