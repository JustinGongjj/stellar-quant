"""行业相对强弱：能否给现有量价特征带来正交增量？

背景
----
判断力诊断（reports/判断力诊断与增量路线.md）的结论是：
现有 19 个特征线性 OOS IC 0.082，加不出新东西；要提高判断力必须加**新类型信息**。

行业相对强弱是最便宜的一种：面板里已有季度行业分类，不需要新数据源，
且逻辑上与个股量价天然正交（个股自己的均线 vs 它所在行业的整体走势）。

构造
----
  ind_mom_20 = 个股过去 20 日收益 − 所属行业中位数收益   （行业相对强弱）
  ind_mom_60 = 同上，60 日窗口
  ind_ret_20 = 所属行业过去 20 日中位数收益              （行业本身动量）

检验口径（与 judgment_increment_test 一致）
------------------------------------------
  1. 单独 IC
  2. 对现有 19 个特征逐日截面回归后的**残差 IC**（正交增量）
  3. 与现有特征的最大相关系数
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path.home() / ".cache" / "stellarq"
R = CACHE / "wide2"
IND = CACHE / "industry"


def load_industry_map():
    """按季度拼出 symbol -> BOARD_NAME 的时间序列，返回 {quarter_start: {sym: board}}"""
    out = {}
    files = sorted(IND.glob("*.parquet"))
    for f in files:
        try:
            d = pd.read_parquet(f)
            q = str(d["REPORTDATE"].iloc[0])[:10]
            out[q] = dict(zip(d["symbol"].astype(str), d["BOARD_NAME"].astype(str)))
        except Exception:
            continue
    return out


def cs_ic(x, y, d, min_n=50):
    df = pd.DataFrame({"d": d, "x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    ics = []
    for _, g in df.groupby("d"):
        if len(g) >= min_n:
            v = g.x.corr(g.y, method="spearman")
            if np.isfinite(v):
                ics.append(v)
    ics = np.array(ics)
    if len(ics) < 10:
        return np.nan, 0
    return float(ics.mean()), len(ics)


def main():
    close = np.load(R / "wide_close.npy", mmap_mode="r")
    dates = np.load(R / "wide_dates.npy", allow_pickle=True)
    syms = np.load(R / "wide_syms.npy", allow_pickle=True)
    nd, ns = close.shape
    print(f"面板 {nd} 日 × {ns} 只", flush=True)

    ind = load_industry_map()
    quarters = sorted(ind.keys())
    print(f"行业快照 {len(quarters)} 个季度（{quarters[0]} → {quarters[-1]}），"
          f"最新 {len(ind[quarters[-1]])} 只 / {len(set(ind[quarters[-1]].values()))} 行业", flush=True)

    sym_str = np.array([str(s) for s in syms])

    # 每个季度给每只股票打上行业编号
    board_of_q = {}
    for q in quarters:
        mp = ind[q]
        codes = [mp.get(s, "") for s in sym_str]
        uniq = {b: i for i, b in enumerate(sorted(set(c for c in codes if c)))}
        board_of_q[q] = np.array([uniq.get(c, -1) for c in codes], dtype=np.int32)

    def board_at(t):
        dq = str(pd.Timestamp(dates[t]))[:10]
        # 用不晚于 t 的最近一个季度
        cand = [q for q in quarters if q <= dq]
        return board_of_q[cand[-1]] if cand else None

    # 抽样交易日构造因子（全部算太慢）
    step = 5
    rows = []
    for t in range(80, nd - 6, step):
        b = board_at(t)
        if b is None:
            continue
        cur = np.asarray(close[t], dtype=np.float64)
        p20 = np.asarray(close[t - 20], dtype=np.float64)
        p60 = np.asarray(close[t - 60], dtype=np.float64) if t >= 60 else None
        ok = np.isfinite(cur) & np.isfinite(p20) & (p20 > 0) & (b >= 0)
        if ok.sum() < 300:
            continue
        r20 = np.full(ns, np.nan)
        r20[ok] = cur[ok] / p20[ok] - 1
        r60 = np.full(ns, np.nan)
        if p60 is not None:
            ok60 = ok & np.isfinite(p60) & (p60 > 0)
            r60[ok60] = cur[ok60] / p60[ok60] - 1

        df = pd.DataFrame({"b": b, "r20": r20, "r60": r60})[ok]
        med20 = df.groupby("b")["r20"].transform("median")
        med60 = df.groupby("b")["r60"].transform("median") if "r60" in df else np.nan

        idx = np.where(ok)[0]
        rows.append(pd.DataFrame({
            "t": t,
            "sym": idx,
            "ind_mom_20": (df["r20"] - med20).to_numpy(),
            "ind_ret_20": med20.to_numpy(),
            "ind_mom_60": (df["r60"] - med60).to_numpy() if isinstance(med60, pd.Series) else np.nan,
        }))

    feat = pd.concat(rows, ignore_index=True)
    print(f"构造 {len(feat)} 行 / {feat.t.nunique()} 个交易日", flush=True)

    # 对齐 v4 的样本（d = 面板日索引，s = 面板股索引）
    X = np.load(CACHE / "v4_X.npy")
    y = np.load(CACHE / "v4_y.npy")
    d = np.load(CACHE / "v4_d.npy")
    s = np.load(CACHE / "v4_s.npy")

    key = feat.set_index(["t", "sym"])
    result = {}
    for col in ("ind_mom_20", "ind_ret_20", "ind_mom_60"):
        vals = []
        for ti, si in zip(d, s):
            try:
                vals.append(key.at[(int(ti), int(si)), col])
            except KeyError:
                vals.append(np.nan)
        x = np.asarray(vals, dtype=float)
        cov = np.isfinite(x).mean()
        ic, n = cs_ic(x, y, d)
        result[col] = (ic, n, cov)
        print(f"\n{col}: 覆盖 {cov*100:.1f}%  单独 IC={ic:+.4f}  n={n}", flush=True)

        # 与现有 19 特征的最大 |相关|
        cors = []
        for k in range(X.shape[1]):
            ok = np.isfinite(x) & np.isfinite(X[:, k])
            if ok.sum() > 1000:
                cors.append(abs(np.corrcoef(x[ok], X[ok, k].astype(float))[0, 1]))
        print(f"  与现有特征最大 |r| = {max(cors):.3f}", flush=True)

        # 正交化
        Z = np.column_stack([X[:, k].astype(float) for k in range(X.shape[1])])
        resid = np.full(len(x), np.nan)
        for day in np.unique(d):
            m = (d == day) & np.isfinite(x) & np.isfinite(Z).all(axis=1)
            if m.sum() < 100:
                continue
            A = np.column_stack([np.ones(m.sum()), Z[m]])
            b = x[m]
            try:
                co, *_ = np.linalg.lstsq(A, b, rcond=None)
                resid[m] = b - A @ co
            except Exception:
                pass
        ic_r, n_r = cs_ic(resid, y, d)
        print(f"  正交化后 IC={ic_r:+.4f}  n={n_r}", flush=True)

    print("\n" + "=" * 60)
    print(f"{'因子':<14}{'单独IC':>9}{'正交后':>9}{'达标?':>8}")
    for k, (ic, n, cov) in result.items():
        # 达标线：正交后 |IC| ≥ 0.015（对应合成提升约 0.2bp，勉强可测）
        ok = "✓" if abs(ic) >= 0.015 else "✗"
        print(f"{k:<14}{ic:+9.4f}{'':>9}{ok:>8}")


if __name__ == "__main__":
    main()