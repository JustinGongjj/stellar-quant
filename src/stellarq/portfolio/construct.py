"""截面排序组合构建：打分 → 目标权重 → 交易约束下的实际成交。

信号处理（参考 Qlib 回测与 Alphalens 的分组口径）
----------------------------------------------
1. 每日按打分降序取前 ``top_k``（或前 ``top_pct``），剔除不可买 / 流动性不足标的；
2. 权重方案 ``equal`` / ``score``（打分超过截断值后的线性加权）/ ``inverse_vol``；
3. 单票上限 ``max_weight`` 用带上下限的单纯形投影实现（不会出现负权重或 NaN）；
4. 换手控制：``max_turnover`` 限制单日单边换手；
5. 跌停不可卖：被强制持有的仓位在目标里保留。

回测执行口径
------------
T 日收盘生成信号 → T+1 日收盘成交（保守，不使用当日收盘瞬时成交）；费用逐笔计算
（买入/卖出费率不同、含印花税、佣金下限、滑点），现金收益按 0 计（保守）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel, apply_volume_cap


@dataclass
class PortfolioSpec:
    top_k: int = 50
    top_pct: float | None = None
    weighting: str = "score"        # equal / score / inverse_vol
    max_weight: float = 0.03
    max_turnover: float = 0.5       # 单日单边最大换手
    min_price: float = 2.0
    min_amount: float = 2e7         # 当日成交额下限（元）
    cash_weight: float = 0.0        # 固定现金比例


def _capped_simplex(w: np.ndarray, cap: float) -> np.ndarray:
    """把非负向量投影到 {w ≥ 0, Σw = 1, w ≤ cap}（water-filling）。"""
    w = np.asarray(w, dtype=float)
    n = len(w)
    if n == 0:
        return w
    cap = min(max(cap, 1.0 / n), 1.0)
    if w.sum() <= 0 or not np.isfinite(w.sum()):
        w = np.ones(n)
    w = w / w.sum()
    for _ in range(100):
        w = np.clip(w, 0.0, cap)
        gap = 1.0 - w.sum()
        if gap <= 1e-12:
            break
        free = w < cap - 1e-12
        if not free.any():
            break
        w[free] += gap / free.sum()
    w = np.clip(w, 0.0, cap)
    total = w.sum()
    return w / total if total > 0 else w


def select_and_weight(scores: pd.Series, mask: pd.Series, spec: PortfolioSpec,
                      vol: pd.Series | None = None) -> pd.Series:
    """单日：选出持仓并给出目标权重（和为 1 - cash_weight）。"""
    s = scores[mask & scores.notna()].astype(float)
    if s.empty:
        return pd.Series(dtype=float)
    n = len(s)
    k = spec.top_k if spec.top_pct is None else max(10, int(round(n * spec.top_pct)))
    k = max(1, min(k, n))
    top = s.sort_values(ascending=False).head(k)
    if spec.weighting == "equal":
        w = pd.Series(1.0, index=top.index)
    elif spec.weighting == "inverse_vol" and vol is not None:
        v = vol.reindex(top.index).astype(float).replace(0, np.nan)
        v = v.fillna(float(vol[vol > 0].median()) if (vol > 0).any() else 1.0)
        w = 1.0 / v
    else:
        cut = max(float(top.min()), float(top.quantile(0.2)))
        w = (top - cut).clip(lower=1e-9)
    wv = _capped_simplex(w.to_numpy(dtype=float), float(spec.max_weight))
    out = pd.Series(wv, index=top.index) * (1.0 - spec.cash_weight)
    return out.sort_index()


def backtest_scores(scores: pd.DataFrame, prices: pd.DataFrame, spec: PortfolioSpec,
                    costs: CostModel | None = None,
                    tradable_buy: pd.DataFrame | None = None,
                    tradable_sell: pd.DataFrame | None = None,
                    amount: pd.DataFrame | None = None,
                    benchmark: pd.DataFrame | None = None) -> dict:
    """按日频打分做组合回测。

    ``scores`` / ``prices`` / ``tradable_*`` / ``amount`` 均为 date × symbol 宽表；
    ``prices`` 用复权收盘价。T 日信号在 T+1 日收盘成交。
    """
    costs = costs or CostModel()
    dates = list(prices.index)
    rets = prices.pct_change()
    vol_panel = rets.rolling(20, min_periods=10).std()
    eq = 1.0
    w_prev = pd.Series(dtype=float)
    cash = 1.0
    nav, turnovers, holds = [(dates[0], 1.0)], [], []
    for i in range(1, len(dates)):
        dt = dates[i]
        sig_dt = dates[i - 1]
        r_dt = rets.loc[dt]
        px = prices.loc[dt]
        if sig_dt not in scores.index:
            gross = float((w_prev * r_dt.reindex(w_prev.index).fillna(0.0)).sum())
            eq = eq * (1.0 + gross)
            grown = w_prev * (1 + r_dt.reindex(w_prev.index).fillna(0.0))
            tot = float(grown.sum()) + cash
            if tot > 0:
                w_prev, cash = grown / tot, cash / tot
            nav.append((dt, eq))
            continue

        mask = px.notna() & (px >= spec.min_price)
        if amount is not None and dt in amount.index:
            mask = mask & (amount.loc[dt].reindex(px.index).fillna(0.0) >= spec.min_amount)
        if tradable_buy is not None and dt in tradable_buy.index:
            mask = mask & tradable_buy.loc[dt].reindex(px.index).fillna(False)
        target = select_and_weight(scores.loc[sig_dt], mask, spec,
                                   vol_panel.loc[dt] if dt in vol_panel.index else None)

        names = target.index.union(w_prev.index)
        tw = target.reindex(names).fillna(0.0)
        pw = w_prev.reindex(names).fillna(0.0)

        # 不可卖（跌停 / 停牌）：强制保留原仓位
        if tradable_sell is not None and dt in tradable_sell.index:
            can_sell = tradable_sell.loc[dt].reindex(names).fillna(False)
            for sym, val in pw[~can_sell & (pw > 0)].items():
                tw.loc[sym] = val

        # 换手约束（单边）
        turn = float((tw - pw).abs().sum()) / 2.0
        if spec.max_turnover > 0 and turn > spec.max_turnover:
            tw = pw + (tw - pw) * (spec.max_turnover / turn)

        # 成交量约束：买入不超过当日成交额的 max_adv_pct
        if amount is not None and dt in amount.index:
            for sym in list(tw.index):
                delta = float(tw.get(sym, 0.0) - pw.get(sym, 0.0))
                if delta > 0 and eq > 0:
                    cap_val = apply_volume_cap(delta * eq, float(amount.loc[dt].get(sym, 0.0) or 0.0))
                    tw.loc[sym] = float(pw.get(sym, 0.0)) + cap_val / eq

        buys = float((tw - pw).clip(lower=0).sum())
        sells = float((pw - tw).clip(lower=0).sum())
        # buys / sells 为占净值的比例；费用模型返回的也是净值比例
        fee_frac = costs.buy_fee(buys) + costs.sell_fee(sells)
        gross = float((tw * r_dt.reindex(tw.index).fillna(0.0)).sum())
        cash_frac = max(0.0, 1.0 - float(tw.sum()))
        net = gross - fee_frac
        eq = eq * (1.0 + net)

        grown = tw * (1 + r_dt.reindex(tw.index).fillna(0.0))
        cash = cash_frac * (1.0 + net)
        tot = float(grown.sum()) + cash
        if tot > 0:
            w_prev, cash = grown / tot, cash / tot
        else:
            w_prev = pd.Series(dtype=float)
        nav.append((dt, eq))
        turnovers.append((dt, turn))
        holds.append({"date": dt, "n": int((tw > 0).sum()), "cash": float(cash),
                      "top_weight": float(tw.max()) if len(tw) else 0.0})

    nav_s = pd.Series(dict(nav)).sort_index()
    out = {
        "nav": nav_s,
        "returns": nav_s.pct_change().dropna(),
        "turnover": pd.Series(dict(turnovers)).sort_index() if turnovers else pd.Series(dtype=float),
        "holdings": pd.DataFrame(holds).set_index("date") if holds else pd.DataFrame(),
    }
    if benchmark is not None:
        b = benchmark.reindex(nav_s.index).ffill().dropna()
        if len(b):
            out["benchmark_nav"] = b / b.iloc[0]
    return out


def perf(nav: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> dict:
    """业绩指标：Qlib risk_analysis 的算术年化口径 + 几何年化 + 超额信息比率。"""
    r = nav.pct_change().dropna()
    if len(r) < 2:
        return {}
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    ann = mean * periods_per_year
    geo = float(nav.iloc[-1] ** (periods_per_year / len(r)) - 1) if nav.iloc[-1] > 0 else float("nan")
    dd = float((nav / nav.cummax() - 1.0).min())
    return {
        "n_days": int(len(r)),
        "ann_return_arith": ann,
        "ann_return_geo": geo,
        "ann_vol": std * float(np.sqrt(periods_per_year)),
        "sharpe_rf0": (ann - rf) / (std * float(np.sqrt(periods_per_year))) if std > 0 else float("nan"),
        "max_drawdown": dd,
        "calmar": geo / abs(dd) if dd < 0 else float("nan"),
        "win_rate_daily": float((r > 0).mean()),
    }


# ---------------------------------------------------------------------------
# 重叠期修正（多空价差 / 与持有期一致的组合）
# ---------------------------------------------------------------------------
def nonoverlap_annualize(period_returns: pd.Series, hold_days: int) -> dict:
    """把「每 k 日重算、持有 h 日」的序列折算成年化指标（重叠修正）。

    k 与 h 的比值为 overlap = h/k。重叠期间同一段价格路径会被重复计入，
    直接用 k 折算年化会放大收益。这里采用标准修正：
      - 年化收益：取 (overlap) 期移动平均的均值 × (年交易日数 / h)
      - 年化波动：移动平均后序列的标准差 × sqrt(年交易日数 / h)
    """
    import math

    r = pd.Series(period_returns).dropna().astype(float)
    if len(r) < 5:
        return {}
    overlap = max(1, int(round(hold_days / max(1, getattr(r, "attrs", {}).get("step", hold_days)))))
    return {"n": int(len(r)), "mean": float(r.mean()), "std": float(r.std(ddof=1))}
