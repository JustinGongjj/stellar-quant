"""绩效口径工具：统一所有策略的年化/波动/回撤计算。

背景（2026-11-14 发现的错误）
------------------------------
早前脚本用「5 期移动平均 × 10.08」把「每 5 交易日一期」的策略年化，
结果把收益和波动**同时压小**（移动平均让波动变成 1/5，均值不变），
年化收益被严重低估（实测 35.40% 被算成 7.26%）。

正确做法：每期持有 h 日、且相邻期首尾相接时，直接链式复利，
年化系数 = 252 / (期数 × h)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualise_periodic(period_returns, hold_days: int = 5, periods_per_year: float | None = None) -> dict:
    """把「每期持有 hold_days 个交易日」的收益序列换算成年化指标。

    Parameters
    ----------
    period_returns : 每期收益（小数），期与期首尾相接、无重叠
    hold_days : 每期覆盖的交易日数
    periods_per_year : 显式指定每年期数（默认 252/hold_days）

    Returns
    -------
    dict: ann_arith（算术年化）、ann_geo（几何年化）、vol、sharpe、max_dd、cum
    """
    r = np.asarray(pd.Series(period_returns).dropna(), dtype=float)
    if r.size == 0:
        return {}
    ppy = periods_per_year if periods_per_year is not None else 252.0 / hold_days
    mu = float(r.mean())
    vol = float(r.std(ddof=1) * np.sqrt(ppy)) if r.size > 1 else float("nan")
    nav = np.cumprod(1.0 + r)
    years = r.size / ppy
    geo = float(nav[-1] ** (1.0 / years) - 1.0) if years > 0 and nav[-1] > 0 else float("nan")
    dd = float((nav / np.maximum.accumulate(nav) - 1.0).min())
    return {
        "n_periods": int(r.size),
        "years": float(years),
        "ann_arith": mu * ppy,
        "ann_geo": geo,
        "vol": vol,
        "sharpe": (mu * ppy) / vol if vol and vol > 0 else float("nan"),
        "max_dd": dd,
        "cum": float(nav[-1]),
        "win_rate": float((r > 0).mean()),
    }


def overlap_adjusted(period_returns, step_days: int = 5, hold_days: int = 5) -> dict:
    """当「调仓步长 < 持有期」（存在重叠）时使用：每 hold/step 期才算 1 个独立观测。

    例：每 5 日调仓、持有 20 日 → 4 期重叠 → 独立观测数 = n/4。
    本仓当前策略 step=hold=5，无重叠，应直接用 ``annualise_periodic``。
    """
    r = pd.Series(period_returns).dropna()
    k = max(1, int(round(hold_days / max(step_days, 1))))
    if k == 1:
        return annualise_periodic(r, hold_days=hold_days)
    # 取每 k 期中的一个，构造不重叠序列
    sub = r.iloc[::k]
    return annualise_periodic(sub, hold_days=hold_days * k)


def non_overlapping_nav(period_returns, hold_days: int = 5) -> pd.Series:
    """返回首尾相接的净值曲线（每期 hold_days 个交易日）。"""
    r = pd.Series(period_returns).dropna()
    return pd.Series(np.cumprod(1.0 + r.to_numpy()), index=r.index)


def self_test() -> None:
    """口径自检：给定已知期均值与波动，验证年化结果符合复利定义。"""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0073, 0.0528, 105)      # 与实测的量级一致
    out = annualise_periodic(r, hold_days=5)
    nav = np.cumprod(1 + r)
    years = 105 * 5 / 252
    assert abs(out["ann_geo"] - (nav[-1] ** (1 / years) - 1)) < 1e-12
    assert out["n_periods"] == 105
    # 关系：对无重叠的独立期收益，复利年化几何 ≈ 算术 − σ²/2（波动拖累），
    # 只有在「期数少、波动大」时两者才可能反号；这里只校验量级一致。
    assert abs(out["ann_geo"] - out["ann_arith"]) < 0.30, (out["ann_geo"], out["ann_arith"])
    print("self_test ok:", {k: round(v, 4) for k, v in out.items() if isinstance(v, float)})


if __name__ == "__main__":
    self_test()
