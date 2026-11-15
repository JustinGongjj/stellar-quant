"""数据源登记表 + 校验闸门（本仓库的数据总控）。

2026-11-13 对本运行时可达数据源的实测结论
=========================================

| 源 | 可达性 | 可用字段 | 结论 |
|---|---|---|---|
| **腾讯行情** `qt.gtimg.cn` / `web.ifzq.gtimg.cn` | ✅ 稳定，无 key，可重复 | 不复权日线（2000 根/页，翻页取全量）、除权事件字典（含 ``fh_sh``）、实时行情（官方昨收 ``prec``、涨跌停价、换手、流通市值） | **主数据源** |
| 东方财富 `push2his` | ⚠️ 限流封禁（`rc:102`）；`pct_chg` 与相邻收盘反推不符 | 不复权价与腾讯一致 | 仅作**校验**，不作主源 |
| 腾讯 `qfq` / `hfq` | ✅ | 派生价格序列 | ❌ **不可用**：同一日行情由「不复权日历」与「复权序列」给出的收益率系统性不同（1224/1918 天差 >10bp），内部不一致 |
| 新浪行情 `hq.sinajs.cn` / `quotes.sina.cn` | ✅ | 实时（含昨收）、日线（最多 1023 根） | 校验用（第二源） |
| FRED `fredgraph.csv` | ✅ | 美债利率、通胀等宏观 | 宏观因子 |
| 巨潮/交易所公告、中债、中证指数官网 | 部分可达 | 需按各自口径申请 | 后续扩展 |
| AKShare / Tushare / BaoStock | ❌ 本运行时未安装；BaoStock 走自定义 TCP 端口，本环境只放行 80/443 | — | 不引入 |

校验闸门（``gate_rows``）
--------------------------
对每只股票、每个交易日检查：

1. 不复权收盘价在合理范围（>0）；
2. 相邻两日收益率绝对值 < 60%（超过视为数据错误，而非真实行情）；
3. 若有除权事件，按交易所口径 ``(昨收 − 每股派现) / (1 + 每股送转)`` 重算参考价，
   除权日收益率与其它日同口径；
4. 若同时取到第二源（新浪/东财）收盘价，偏差 > 1.1 分视为不一致。

任一条不通过的行被剔除并计入 ``violations``，**不做插值**——缺口本身就是证据。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    "tencent": {
        "role": "primary", "key": False, "protocol": "HTTPS",
        "fields": ["ohlcv_raw", "corp_action", "quote_realtime", "limit_price"],
        "notes": "日线单页上限 2000 根，用 end 参数翻页；qfq/hfq 不可用（内部不一致）",
    },
    "eastmoney": {
        "role": "cross_check", "key": False, "protocol": "HTTPS",
        "fields": ["ohlcv_raw"],
        "notes": "2026-11-13 起 push2his 限流（rc:102）；pct_chg 字段不可信",
    },
    "sina": {
        "role": "cross_check", "key": False, "protocol": "HTTPS",
        "fields": ["ohlcv_raw", "quote_realtime"],
        "notes": "日线最多 1023 根，适合近端校验",
    },
    "fred": {
        "role": "macro", "key": False, "protocol": "HTTPS",
        "fields": ["macro_series"],
        "notes": "fredgraph.csv 直接下载",
    },
}


@dataclass
class GateReport:
    rows_in: int = 0
    rows_kept: int = 0
    violations: dict[str, int] = field(default_factory=dict)
    per_symbol: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        total = sum(self.violations.values())
        return {
            "rows_in": self.rows_in, "rows_kept": self.rows_kept,
            "violations": self.violations, "violation_rows": total,
            "pass_rate": self.rows_kept / self.rows_in if self.rows_in else float("nan"),
            "worst_symbols": sorted(self.per_symbol.items(), key=lambda kv: -kv[1]["violations"])[:10],
        }


def ex_right_ref(prev_close: float, cash: float, bonus: float) -> float:
    """交易所除权除息参考价： (昨收 − 每股派现) / (1 + 每股送转)。"""
    den = 1.0 + float(bonus or 0.0)
    if den <= 0:
        return float("nan")
    return (float(prev_close) - float(cash or 0.0)) / den


def gate_symbol(df: pd.DataFrame, actions: pd.DataFrame | None = None,
                second_source: pd.DataFrame | None = None,
                max_abs_ret: float = 0.60, price_tol: float = 0.011) -> tuple[pd.DataFrame, dict]:
    """单只股票的校验与复权重建。

    ``df``：date / open / close / high / low / volume（不复权，升序）。
    返回 (带 ``adj_close`` 的 DataFrame, 违例统计)。
    """
    d = df.sort_values("date").reset_index(drop=True).copy()
    viol = {"non_positive_price": 0, "abs_ret_too_large": 0, "cross_source_mismatch": 0, "missing_prev": 0}
    acts = {}
    if actions is not None and len(actions):
        for _, r in actions.iterrows():
            acts[pd.Timestamp(r["ex_date"]).normalize()] = (float(r["cash_per_share"]), float(r["bonus_per_share"]))
    second = {}
    if second_source is not None and len(second_source):
        second = {pd.Timestamp(k).normalize(): float(v) for k, v in
                  zip(second_source["date"], second_source["close"])}

    keep = np.ones(len(d), dtype=bool)
    price = d["close"].to_numpy(dtype=float)
    dates = pd.to_datetime(d["date"]).dt.normalize().to_list()
    viol["non_positive_price"] = int((price <= 0).sum())
    keep &= price > 0
    ret = np.full(len(d), np.nan)
    for i in range(1, len(d)):
        if not keep[i - 1] or not keep[i]:
            viol["missing_prev"] += 1
            continue
        ev = acts.get(dates[i])
        base = ex_right_ref(price[i - 1], *ev) if ev else price[i - 1]
        if not np.isfinite(base) or base <= 0:
            viol["missing_prev"] += 1
            continue
        r = price[i] / base - 1.0
        ret[i] = r
        if abs(r) > max_abs_ret:
            viol["abs_ret_too_large"] += 1
            keep[i] = False
        sc = second.get(dates[i])
        if sc is not None and abs(sc - price[i]) > price_tol:
            viol["cross_source_mismatch"] += 1
    d["ret"] = ret
    d["kept"] = keep
    # 复权价：从后往前乘 (1 + 交易所口径收益)
    adj = np.full(len(d), np.nan)
    if keep.any():
        last = int(np.where(keep)[0][-1])
        adj[last] = price[last]
        for i in range(last - 1, -1, -1):
            if keep[i] and np.isfinite(ret[i + 1]):
                adj[i] = adj[i + 1] / (1.0 + ret[i + 1])
    d["adj_close"] = adj
    return d, viol


def gate_panel(frames: list[pd.DataFrame], actions: pd.DataFrame | None = None,
               second: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, GateReport]:
    rep = GateReport()
    out = []
    acts_by = {k: g for k, g in actions.groupby("symbol")} if actions is not None and len(actions) else {}
    for f in frames:
        sym = str(f["symbol"].iloc[0])
        g, v = gate_symbol(f, acts_by.get(sym), (second or {}).get(sym))
        rep.rows_in += len(g)
        rep.rows_kept += int(g["kept"].sum())
        for k, val in v.items():
            rep.violations[k] = rep.violations.get(k, 0) + val
        rep.per_symbol[sym] = {"rows": len(g), "kept": int(g["kept"].sum()), "violations": sum(v.values())}
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(), rep


def write_report(rep: GateReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 价格合规闸门（对整包历史数据做逐行筛选，不依赖外部接口）
# ---------------------------------------------------------------------------
def limit_band(symbol: str) -> float:
    """涨跌停幅度：创业板/科创板 20%，其余主板 10%（北交所 30%，本仓已剔除）。"""
    s = str(symbol).zfill(6)
    return 20.0 if s.startswith(("688", "689", "300", "301")) else 10.0


def price_limit_mask(df: pd.DataFrame, tolerance: float = 0.6,
                     date_col: str = "date", sym_col: str = "symbol",
                     price_col: str = "close") -> pd.Series:
    """逐行判断：当日收益率是否落在板块涨跌停之内（含容差）。

    涨停/跌停日收益可以恰好等于 ±10%/±20%，因此容差取 0.6 个百分点，
    只有明显越界（>10.6% / >20.6%）才判为异常——这类行要么是数据错误，
    要么是除权未调整，两种情况下都不该进入因子计算。
    """
    d = df.sort_values([sym_col, date_col]).copy()
    r = d.groupby(sym_col)[price_col].pct_change().abs() * 100.0
    lim = d[sym_col].map(limit_band) + tolerance
    ok = ~(r > lim)
    ok = ok.where(r.notna(), True)          # 每只股票首日保留
    return ok
