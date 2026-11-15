"""A 股交易成本与成交约束（回测必须满足，否则收益不可信）。

吸收自通行做法与 Qlib 回测器的约束设计：

* **印花税**：卖出单边 0.05%（2023-08-28 起；之前 0.1%），买入不收；
* **佣金**：双边，默认万分之 1.5，最低 5 元/笔；**过户费**：双边 0.001%；
* **涨跌停**：主板 ±10%，创业板/科创板 ±20%，北交所 ±30%，ST ±5%；
  涨停时不能买（买不到），跌停时不能卖（卖不出）；
* **停牌**：无成交，不能交易；
* **T+1**：当日买入的股票当日不能卖出（按日频回测时意味着持仓至少隔一日）；
* **成交量约束**：单日成交额不超过该股成交额的 ``max_adv_pct``（默认 5%），否则按比例成交。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class CostModel:
    stamp_tax_sell: float = 0.0005      # 印花税（仅卖出）
    commission: float = 0.00015         # 佣金（双边）
    min_commission: float = 5.0         # 最低佣金（元/笔）
    transfer_fee: float = 0.00001       # 过户费（双边）
    slippage_bps: float = 5.0           # 滑点（万分之几）

    def buy_cost_rate(self) -> float:
        return self.commission + self.transfer_fee + self.slippage_bps / 10000.0

    def sell_cost_rate(self) -> float:
        return self.commission + self.transfer_fee + self.slippage_bps / 10000.0 + self.stamp_tax_sell

    # 接口约定：notional 为「占组合净值的比例」（回测里逐日只有净值比例，没有股数）。
    # 最低佣金按「每笔 5 元 / 组合规模」折算，组合规模由调用方传入；
    # 未传规模时按 100 万元估算（1% 量级误差不影响结论，但必须写在 manifest 里）。
    portfolio_value: float = 1_000_000.0

    def buy_fee(self, notional_frac: float) -> float:
        """返回占净值的费用比例。"""
        if notional_frac <= 0:
            return 0.0
        notional = notional_frac * self.portfolio_value
        fee = max(notional * self.commission, self.min_commission) + notional * (self.transfer_fee + self.slippage_bps / 10000.0)
        return fee / self.portfolio_value

    def sell_fee(self, notional_frac: float) -> float:
        if notional_frac <= 0:
            return 0.0
        notional = notional_frac * self.portfolio_value
        fee = (max(notional * self.commission, self.min_commission)
               + notional * (self.transfer_fee + self.stamp_tax_sell + self.slippage_bps / 10000.0))
        return fee / self.portfolio_value


def stamp_tax_rate(date: pd.Timestamp | str) -> float:
    """印花税随政策变化：2023-08-28 起减半为 0.05%。"""
    ts = pd.Timestamp(date)
    return 0.0005 if ts >= pd.Timestamp("2023-08-28") else 0.001


def limit_ratio(limit_band: str, is_st: bool = False) -> float:
    if is_st:
        return 0.05
    return {"MAIN": 0.10, "CHINEXT": 0.20, "STAR": 0.20, "BSE": 0.30}.get(str(limit_band), 0.10)


def tradable_today(row: dict) -> tuple[bool, bool]:
    """返回 (可买, 可卖)。row 需包含 close / prev_close / limit_band / suspended / is_st。"""
    if row.get("suspended", False):
        return False, False
    px = float(row.get("close") or 0.0)
    prev = float(row.get("prev_close") or 0.0)
    if px <= 0 or prev <= 0:
        return False, False
    limit = limit_ratio(row.get("limit_band", "MAIN"), bool(row.get("is_st", False)))
    upper = round(prev * (1 + limit) + 1e-9, 2)
    lower = round(prev * (1 - limit) + 1e-9, 2)
    buy_ok = px < upper - 1e-9
    sell_ok = px > lower + 1e-9
    return buy_ok, sell_ok


def apply_volume_cap(order_value: float, amount: float, max_adv_pct: float = 0.05) -> float:
    """成交量约束：单日成交额不超过该股当日成交额的 ``max_adv_pct``。"""
    cap = float(amount) * max_adv_pct
    if not math.isfinite(cap) or cap <= 0:
        return 0.0
    return min(order_value, cap)
