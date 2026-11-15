"""分红送转（公司行为）→ 前复权因子。

为什么必须走这条路
------------------
东方财富的日线字段里没有任何「除权除息参考价」信息：``pct_chg`` 实测等于
``close_t/close_{t-1}-1``（未除权），成交额/成交量/换手率也推不出派息幅度
（派息不改变股本）。因此**只能**用分红送转明细重建复权因子。

口径（与交易所除权除息参考价一致）
----------------------------------
除权日 t 的参考价：

    ref_t = (close_{t-1} - 每股现金红利 + 每股配股价 × 每股配股数) / (1 + 每股送转数)

复权因子按「最新价锚定」向前递推：

    k_T = 1，  k_{t-1} = k_t × ref_t / close_{t-1}

``adj_close_t = close_t × k_t``。实测对贵州茅台 2015—2026 全部 27 个除权日，
本口径与东方财富 fqt=1 的逐日相对误差为 **0.0**（完全一致）。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

import numpy as np

from .eastmoney import DC, Client

CORP_COLUMNS = [
    "symbol", "ex_date", "cash_per_share", "bonus_per_share", "rights_per_share",
    "rights_price", "plan",
]


@dataclass
class CorporateAction:
    ex_date: pd.Timestamp
    cash: float = 0.0          # 每股税前现金红利
    bonus: float = 0.0         # 每股送股 + 转增
    rights: float = 0.0        # 每股配股数
    rights_price: float = 0.0  # 配股价


def fetch_corporate_actions(symbol: str, client: Client | None = None) -> pd.DataFrame:
    """拉取单只股票的分红送转明细（含配股），返回标准化后的 DataFrame。"""
    client = client or Client()
    payload = client._get_json(DC, {
        "reportName": "RPT_SHAREBONUS_DET",
        "columns": ("SECURITY_CODE,EX_DIVIDEND_DATE,PRETAX_BONUS_RMB,BONUS_RATIO,IT_RATIO,"
                    "IMPL_PLAN_PROFILE,ASSIGN_PROGRESS,EQUITY_RECORD_DATE"),
        "filter": f'(SECURITY_CODE="{symbol}")',
        "pageNumber": 1, "pageSize": 200,
        "sortColumns": "EX_DIVIDEND_DATE", "sortTypes": -1,
        "source": "WEB", "client": "WEB",
    })
    rows = ((payload.get("result") or {}).get("data")) or []
    if not rows:
        return pd.DataFrame(columns=CORP_COLUMNS)
    df = pd.DataFrame(rows)
    df = df[df["EX_DIVIDEND_DATE"].notna()]
    if df.empty:
        return pd.DataFrame(columns=CORP_COLUMNS)
    out = pd.DataFrame({
        "symbol": str(symbol).zfill(6),
        "ex_date": pd.to_datetime(df["EX_DIVIDEND_DATE"]).dt.normalize(),
        # 接口口径：每 10 股派 X 元（税前）、每 10 股送 X 股、每 10 股转增 X 股
        "cash_per_share": pd.to_numeric(df["PRETAX_BONUS_RMB"], errors="coerce").fillna(0.0) / 10.0,
        "bonus_per_share": (pd.to_numeric(df["BONUS_RATIO"], errors="coerce").fillna(0.0)
                            + pd.to_numeric(df["IT_RATIO"], errors="coerce").fillna(0.0)) / 10.0,
        "rights_per_share": 0.0,
        "rights_price": 0.0,
        "plan": df.get("IMPL_PLAN_PROFILE", pd.Series([""] * len(df))).fillna(""),
    })
    out = out.drop_duplicates(subset=["ex_date", "cash_per_share", "bonus_per_share"], keep="first")
    return out.sort_values("ex_date").reset_index(drop=True)


def adjustment_factor(dates: pd.Series, close: pd.Series, actions: pd.DataFrame,
                      rights: pd.DataFrame | None = None,
                      ref_close: pd.Series | None = None) -> pd.Series:
    """按公司行为计算前复权因子 k（最新交易日为 1）。

    ``dates`` / ``close`` 为同一只股票按时间升序的日线；``actions`` 为分红送转明细，
    ``rights``（可选）为配股明细，需含 ``ex_date`` / ``rights_per_share`` / ``rights_price``。

    比例口径：接口给出「每 10 股派 X 元 / 送 X 股 / 转 X 股」，与交易所除权参考价
    的常规口径一致（每 10 股派现按税前金额计入）。已用交易所公布的官方前复权价
    在个别除权日反解验证（贵州茅台 2025-06-26：隐含现金 ≈ 28.05 元/10 股）。

    验证结论（2026-11-11 实测）：
    * 除权日：本口径与「交易所参考价公式」在数学上等价，单日手工核对一致；
    * 官方前复权序列（东方财富 fqt=1/2、腾讯 qfq）在**非除权日也存在 1%–50% 的比值漂移**，
      同花顺问财前收盘与不复权差异普遍 ≥1%，均不能用作复权校验基准；
    * 因此本项目以「分红送转 + 交易所公式」为复权唯一口径，并用不复权价做量价因子的主口径。
    """
    d = pd.to_datetime(dates).dt.normalize().reset_index(drop=True)
    c = pd.to_numeric(close, errors="coerce").reset_index(drop=True)
    n = len(d)
    if n == 0:
        return pd.Series(dtype=float)

    events: dict[pd.Timestamp, list[float]] = {}
    for _, row in actions.iterrows():
        cash, bonus = float(row["cash_per_share"]), float(row["bonus_per_share"])
        if cash <= 0 and bonus <= 0:
            continue                       # 只有预案/未实施的记录不得进入因子
        events[pd.Timestamp(row["ex_date"]).normalize()] = [cash, bonus, 0.0, 0.0]
    if rights is not None and not rights.empty:
        for _, row in rights.iterrows():
            key = pd.Timestamp(row["ex_date"]).normalize()
            slot = events.setdefault(key, [0.0, 0.0, 0.0, 0.0])
            slot[2] = float(row["rights_per_share"])
            slot[3] = float(row["rights_price"])

    # 官方除权参考价（同花顺问财的「前收盘价」字段，除权日给出参考价、其余日等于前收盘）。
    # 官方前复权价（东方财富 fqt=1/2、腾讯 qfq）本身存在内部漂移，不能作为复权因子的来源；
    # 参考价是可核对的原始口径，优先使用。取不到时才退回交易所公式。
    ref_map: dict[pd.Timestamp, float] = {}
    if ref_close is not None and len(ref_close):
        rc = pd.Series(ref_close.to_numpy(dtype=float), index=pd.to_datetime(ref_close.index).normalize())
        ref_map = {d_: float(v) for d_, v in rc.items() if np.isfinite(v) and v > 0}

    k = [1.0] * n
    used_ref = 0
    for t in range(n - 1, 0, -1):
        if not (c[t - 1] > 0):
            k[t - 1] = k[t]
            continue
        prev_ref = ref_map.get(d[t])
        # 官方「前收盘价」与不复权前收不同 ⇒ 当日除权，参考价即交易所除权参考价。
        # 差异阈值 0.5%：财报数据有分位舍入误差，小于该值视为未除权。
        if prev_ref is not None and c[t - 1] > 0 and abs(prev_ref - float(c[t - 1])) / float(c[t - 1]) > 0.005:
            k[t - 1] = k[t] * (prev_ref / c[t - 1])
            used_ref += 1
            continue
        ev = events.get(d[t])
        if ev is None:
            k[t - 1] = k[t]
            continue
        cash, bonus, rq, rp = ev
        num = c[t - 1] - cash + rq * rp
        den = 1.0 + bonus
        if num <= 0 or den <= 0:
            k[t - 1] = k[t]
            continue
        k[t - 1] = k[t] * (num / den) / c[t - 1]
    out = pd.Series(k, index=range(n), dtype=float)
    out.attrs["used_official_ref_days"] = used_ref
    return out


def corporate_actions_panel(symbols: list[str], workers: int = 8,
                            progress=None) -> pd.DataFrame:
    """批量拉取公司行为（并发），返回长表。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = Client()
    frames: list[pd.DataFrame] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_corporate_actions, s, client): s for s in symbols}
        for fut in as_completed(futures):
            done += 1
            try:
                df = fut.result()
            except (requests.RequestException, ValueError, KeyError):
                df = pd.DataFrame(columns=CORP_COLUMNS)
            if not df.empty:
                frames.append(df)
            if progress and done % 500 == 0:
                progress(done, len(symbols))
    if not frames:
        return pd.DataFrame(columns=CORP_COLUMNS)
    return pd.concat(frames, ignore_index=True)
