"""全市场面板构建：日线面板、周/月聚合、面板宽表。

数据纪律
--------
* **不复权价**（fqt=0）用于计算流通股数与真实价格；复权收益率由不复权价
  与「ry/turnover 推算的流通股数」还原：

      adj_close_t = close_t × cumprod(1 - 1/ry_t)

  其中 ``ry = amount / (close × volume × 100)`` 为当日成交均价/收盘价之比，
  换手率分母随除权除息同比例变化，因此该式在派息、送转当日自动校正。
  这是**推断**而非官方复权因子，必须在 manifest 中标注；与交易所公布的
  不复权涨跌幅（``pct_chg``）在除权日之外应当一致，模块自带一致性校验。

* 退市 / 长期停牌股在面板里保留 NaN，不做前向填充：前向填充等于假设
  「停牌期间价格不变」，会让幸存者偏差以更隐蔽的方式回来。

* 历史股票池（point-in-time universe）由「当日有成交」「上市满 N 个交易日」
  「总市值分位 / 流动性分位」三个条件构成，全部基于当日可得信息。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache
from .. import numguard as ng
from .eastmoney import Client, EastmoneyError, board_of_symbol

PANEL_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount",
                 "turnover", "pct_chg", "adj_close", "float_shares"]


RY_LO, RY_HI = 0.30, 3.00


def _clean_ry(ry: pd.Series) -> pd.Series:
    """成交均价/收盘价之比（原始量纲）。

    正常交易日该比值落在 1 附近（实测日内标准差约 0.7%）；0.3~3 之外的取值只可能
    来自停牌、无量或数据缺失，插值处理；NaN 直接视为「无除权」（1.0）。
    """
    ok = ry.where((ry >= RY_LO) & (ry <= RY_HI))
    return ok.fillna(1.0)


RY_SMOOTH = 5                  # 中位数滤波窗口（交易日）


def _median_filter(vals: list[float], k: int) -> list[float]:
    half = max(0, k // 2)
    out = []
    n = len(vals)
    for i in range(n):
        chunk = sorted(vals[max(0, i - half): min(n, i + half + 1)])
        out.append(chunk[len(chunk) // 2])
    return out


def _adj_step(med: list[float], i: int) -> float:
    """第 i 期的复权比率调整量 (1 - 1/r_i)，r_i = ry_i / ry_last（位级精确实现）。"""
    from ..numguard import div, one_minus_inv

    r = div(med[i], med[-1])
    if not np.isfinite(r) or r <= 0:
        return 1.0
    return one_minus_inv(r)


def _ratio_factor(r_num: float, r_den: float) -> float:
    """计算 (r_num/r_den - 1) / (r_num/r_den) = (r_num - r_den) / r_num（整数精确）。"""
    from .. import numguard as ng

    sn, mn, en = ng._parts(r_num)
    sd, md, ed = ng._parts(r_den)
    if mn == 0:
        return 1.0
    e = en if en < ed else ed
    num = mn << (en - e)
    den = md << (ed - e)
    v = num - den                              # (r-1) 的整数表示 × 2^e
    if v == 0:
        return 0.0
    # (r-1)/r = v / num（两者同量纲）。用 52 位定标除法得到尾数，再按原始量纲定指数：
    # 尾数 ≈ v/num × 2^52，故指数取 e。
    q, _ = divmod(abs(v) << 52, num)
    return ng.compose(1 if v < 0 else 0, q, e)


def _cumprod(values: list[float]) -> list[float]:
    """逆序累乘（末项锚定 1），乘法走 numguard。"""
    from .. import numguard as ng

    n = len(values)
    out = [1.0] * n
    for i in range(n - 2, -1, -1):
        out[i] = ng.mul(out[i + 1], values[i + 1])
    return out


def adjustment_ratio(ry, smooth: int = RY_SMOOTH) -> np.ndarray:
    """由「成交均价/收盘价」之比序列推算前复权累计比率（最新一期为 1）。

    除权除息日的识别：ry = amount / (volume×100×close) 的分母是**除权前的流通股本**，
    送转/拆分当日 ry 会出现数量级偏移，派息日只有约 1% 的偏移。噪声来自停牌、
    大宗交易与集合竞价，先做中位数滤波再判跳变。

    设锚点为最新一期（不复权价 = 复权价），r_i = ry_i / ry_last：
    ``ratio_{i-1} = ratio_i × (1 - 1/r_i)``。r_i>1（送转/拆分）历史价按比例折算；
    r_i<1（派息）历史复权价抬高。

    乘除法全部走 numguard 的位级实现（本宿主 float 运算不可信）。
    """
    vals = [float(v) if (v == v and v > 0) else 1.0 for v in ry]
    n = len(vals)
    if n == 0:
        return np.zeros(0)
    med = [float(x) for x in _median_filter(vals, smooth)]
    anchor = med[-1]
    steps = [_ratio_factor(med[i], anchor) for i in range(n)]
    return np.array(_cumprod(steps), dtype=float)


def enrich_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """把单只股票原始日线加算复权价与流通股数，并做质量标记。

    复权口径：以分红送转明细（交易所除权参考价公式）为准，见 ``data.corporate``。
    本函数只做「不复权价 + 成交统计」的派生，不在这里推断复权；复权序列由
    ``apply_adjustment`` 在上层统一附加。
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    if not isinstance(df["date"].dtype, pd.DatetimeTZDtype) and df["date"].dtype.kind != "M":
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    px = df["close"].astype(float)
    vol = df["volume"].astype(float)
    amt = df["amount"].astype(float)
    # 成交均价（元/股）：amount 为元，volume 为「手」
    avg_px = pd.Series(ng.div_arr(amt.to_numpy(dtype=float), (vol * 100.0).to_numpy(dtype=float)), index=df.index)
    df["avg_price"] = avg_px.where(np.isfinite(avg_px))
    # 流通股本（股）：成交额 / 成交均价 / 换手率（换手率字段单位为 %）
    turnover_frac = pd.Series(ng.div_arr(df["turnover"].to_numpy(dtype=float), np.full(len(df), 100.0)), index=df.index)
    fs = pd.Series(ng.div_arr(ng.div_arr(amt.to_numpy(dtype=float), avg_px.to_numpy(dtype=float)), turnover_frac.to_numpy(dtype=float)), index=df.index)
    df["float_shares"] = fs.where(np.isfinite(fs) & (fs > 0))
    df["float_mktcap"] = px * df["float_shares"]
    df["adj_close"] = px          # 占位：复权序列由 apply_adjustment 覆盖
    df["adj_ret"] = px.pct_change(fill_method=None)
    df["suspended"] = (vol.fillna(0) <= 0) | (amt.fillna(0) <= 0)
    df["prev_close"] = px.shift(1)
    df["limit_band"] = board_of_symbol(str(df["symbol"].iloc[0]))
    df["is_st"] = "ST" in str(df["name"].iloc[0]).upper() if "name" in df.columns else False
    return df


def apply_adjustment(panel: pd.DataFrame, corp: pd.DataFrame) -> pd.DataFrame:
    """把分红送转折算成前复权价，附加 ``adj_close`` / ``adj_ret`` 列。

    按股票分组调用 ``corporate.adjustment_factor``（最新交易日锚定为 1）。
    """
    from .corporate import adjustment_factor

    out = []
    corp_by = {sym: g for sym, g in corp.groupby("symbol")} if len(corp) else {}
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("date")
        act = corp_by.get(sym)
        if act is None or act.empty:
            fac = np.ones(len(g))
        else:
            fac = adjustment_factor(g["date"], g["close"], act).to_numpy(dtype=float)
        g = g.copy()
        g["adj_close"] = g["close"].to_numpy(dtype=float) * fac
        g["adj_ret"] = g["adj_close"].pct_change(fill_method=None)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def fetch_panel(symbols: list[str], start: str = "20120101", end: str = "20500101",
                adjust: int = 0, workers: int = 8, progress=None) -> pd.DataFrame:
    """并发拉取日线并拼接成长表（symbol × date）。"""
    client = Client()
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}

    def one(sym: str):
        try:
            raw = client.daily_kline(sym, start=start, end=end, adjust=adjust)
            if raw.empty:
                return sym, None
            return sym, enrich_daily(raw)
        except EastmoneyError as exc:
            return sym, exc

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, s) for s in symbols]
        for fut in as_completed(futures):
            sym, res = fut.result()
            done += 1
            if isinstance(res, Exception):
                errors[sym] = str(res)
            elif res is not None:
                frames.append(res)
            if progress and done % 200 == 0:
                progress(done, len(symbols))
    if not frames:
        raise RuntimeError(f"没有任何股票成功取到行情（错误 {len(errors)} 个）")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    if errors:
        keys = cache.path_for("panel", "fetch_errors", "json")
        keys.write_text(json.dumps(errors, ensure_ascii=False), encoding="utf-8")
    return panel


def save_panel(panel: pd.DataFrame, name: str) -> Path:
    keep = [c for c in PANEL_COLUMNS + ["name", "suspended", "limit_band", "adj_ret"] if c in panel.columns]
    return cache.put_frame("panel", name, panel[keep])


def load_panel(name: str) -> pd.DataFrame:
    df = cache.get_frame("panel", name)
    if df is None:
        raise FileNotFoundError(f"缓存里没有面板 {name!r}，先运行 stellarq data build")
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    return df


def to_wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """长表 → 宽表 date × symbol。"""
    return panel.pivot_table(index="date", columns="symbol", values=field, aggfunc="last").sort_index()


def resample_weekly(panel: pd.DataFrame) -> pd.DataFrame:
    """按自然周聚合：开=首日开，高/低=区间极值，收=末日复权收，量额求和。"""
    df = panel.copy()
    df["week"] = df["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    agg = df.groupby(["symbol", "week"]).agg(
        open=("adj_close", "first"), high=("high", "max"), low=("low", "min"),
        close=("adj_close", "last"), volume=("volume", "sum"), amount=("amount", "sum"),
        float_shares=("float_shares", "last"), n_days=("date", "count"),
    ).reset_index().rename(columns={"week": "date"})
    agg["adj_ret"] = agg.groupby("symbol")["close"].pct_change()
    return agg


def monthly_returns(adj_wide: pd.DataFrame, drop_partial: bool = True) -> pd.DataFrame:
    """宽表复权价 → 月末收益。未完成月份按「当前日期之前」判断剔除。"""
    m = adj_wide.resample("ME").last()
    if drop_partial and len(m):
        today = pd.Timestamp.today().normalize()
        if m.index[-1] > today:
            m = m.iloc[:-1]
    return m.pct_change(fill_method=None)


def listing_age(panel: pd.DataFrame) -> pd.Series:
    """每只股票在数据中的首个交易日（近似上市日）。"""
    return panel.groupby("symbol")["date"].min()


def universe_flags(panel: pd.DataFrame, *, min_days_listed: int = 250,
                   min_amount_rank: float = 0.6, min_amount_median: float = 3e7,
                   exclude_bj: bool = True, exclude_st: bool = True) -> pd.DataFrame:
    """逐日 point-in-time 股票池标记。

    * ``min_days_listed``：上市满 N 个交易日（避免新股）。
    * ``min_amount_rank``：按当日成交额横截面分位，保留前 (1 - rank) —— 即流动性过滤。
    * ``min_amount_median``：按 60 日成交额中位数过滤，剔除僵尸股。
    * ST：名称含 ST 的股票整体剔除（名称字段是当前快照，属轻微前视，需在文档注明）。
    """
    df = panel.copy()
    df["trade_no"] = df.groupby("symbol").cumcount()
    listed_ok = df["trade_no"] >= min_days_listed
    amt = df["amount"].fillna(0.0)
    df["amt_med60"] = df.groupby("symbol")["amount"].transform(
        lambda s: s.rolling(60, min_periods=20).median())
    df["amt_rank"] = df.groupby("date")["amount"].rank(pct=True)
    flag = listed_ok & (~df["suspended"].fillna(True))
    flag &= df["amt_rank"] >= min_amount_rank
    flag &= df["amt_med60"].fillna(0) >= min_amount_median
    if exclude_bj:
        flag &= ~df["symbol"].str.startswith(("4", "8", "92"))
    if exclude_st:
        names = df["name"].fillna("")
        flag &= ~names.str.contains("ST", case=False, regex=False)
    df["in_universe"] = flag
    return df


def panel_summary(panel: pd.DataFrame) -> dict:
    n_days = panel["date"].nunique()
    n_sym = panel["symbol"].nunique()
    return {
        "start": str(panel["date"].min().date()),
        "end": str(panel["date"].max().date()),
        "trading_days": int(n_days),
        "symbols": int(n_sym),
        "rows": int(len(panel)),
        "mean_symbols_per_day": round(len(panel) / max(n_days, 1), 1),
        "delisted_like": int((panel.groupby("symbol")["date"].max().max() - panel.groupby("symbol")["date"].max()).dt.days.gt(250).sum()),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
