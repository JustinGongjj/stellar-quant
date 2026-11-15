"""从行情面板构建宽表矩阵（date × symbol），带价格合规过滤（两遍法，内存可控）。

为什么分两遍：本环境 cgroup 上限 2 GB，把 1,300 万行的多个字段一次性 concat 必然 OOM。
第一遍只读 date/symbol，建立轴索引并写出有效样本掩码；
第二遍逐字段读一列、一次性 scatter 进矩阵并落盘。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..data.sources import price_limit_mask

FIELDS = ["open", "high", "low", "close", "volume", "amount", "turnover", "float_mktcap", "avg_price"]
AXES = ["symbol", "date"]


def _valid_mask(d: pd.DataFrame, min_amount: float, exclude_st: bool) -> np.ndarray:
    ok = np.array(price_limit_mask(d).to_numpy(), dtype=bool, copy=True)
    ok &= (d["close"].to_numpy(dtype=float) > 0)
    ok &= (np.nan_to_num(d["amount"].to_numpy(dtype=float)) >= min_amount)
    if exclude_st:
        ok &= ~d["name"].astype(str).str.contains("ST", regex=False).to_numpy()
    return ok


def build(panel_path: str | Path, out_dir: str | Path, cutoff: str | None = None,
          min_amount: float = 2.0e7, exclude_st: bool = True) -> dict:
    panel_path, out_dir = Path(panel_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(panel_path))
    n_rg = pf.num_row_groups

    # ---------- 第一遍：轴 + 掩码（只保留合规行）
    di_parts, si_parts = [], []
    dates_all, syms_all = set(), set()
    rows_in = rows_kept = 0
    for i in range(n_rg):
        d = pf.read_row_group(i, columns=AXES + ["close", "amount", "name"]).to_pandas()
        ok = _valid_mask(d, min_amount, exclude_st)
        if cutoff:
            ok &= (d["date"] <= np.datetime64(cutoff)).to_numpy()
        rows_in += len(d)
        if ok.any():
            dd = d.loc[ok, AXES]
            rows_kept += len(dd)
            dates_all.update(pd.to_datetime(dd["date"]).unique())
            syms_all.update(dd["symbol"].astype(str).unique())
        di_parts.append(np.where(ok, 0, -1), )
        del d
        print("pass1 rg", i, int(ok.sum()), flush=True)
    dates = np.array(sorted(dates_all))
    syms = np.array(sorted(syms_all))
    dmap = {d: i for i, d in enumerate(dates)}
    smap = {s: i for i, s in enumerate(syms)}
    nd, ns = len(dates), len(syms)
    valid = np.zeros((nd, ns), dtype=bool)
    di_all, si_all = [], []
    for i in range(n_rg):
        d = pf.read_row_group(i, columns=AXES + ["close", "amount", "name"]).to_pandas()
        ok = _valid_mask(d, min_amount, exclude_st)
        if cutoff:
            ok &= (d["date"] <= np.datetime64(cutoff)).to_numpy()
        dd = d.loc[ok]
        if len(dd):
            di_all.append(pd.to_datetime(dd["date"]).map(dmap).to_numpy(dtype=np.int32))
            si_all.append(dd["symbol"].astype(str).map(smap).to_numpy(dtype=np.int32))
        del d
    di = np.concatenate(di_all) if di_all else np.array([], dtype=np.int32)
    si = np.concatenate(si_all) if si_all else np.array([], dtype=np.int32)
    valid[di, si] = True
    np.save(out_dir / "wide_dates.npy", dates)
    np.save(out_dir / "wide_syms.npy", syms)
    np.save(out_dir / "wide_valid.npy", valid)
    del di_all, si_all
    print("axes", nd, ns, "rows kept", rows_kept, flush=True)

    # ---------- 第二遍：逐字段散列
    for f in FIELDS:
        vals = []
        for i in range(n_rg):
            d = pf.read_row_group(i, columns=AXES + [f, "close", "amount", "name"]).to_pandas()
            ok = _valid_mask(d, min_amount, exclude_st)
            if cutoff:
                ok &= (d["date"] <= np.datetime64(cutoff)).to_numpy()
            vals.append(d.loc[ok, f].to_numpy(dtype=np.float32))
            del d
        v = np.concatenate(vals)
        del vals
        m = np.full((nd, ns), np.nan, dtype=np.float32)
        m[di, si] = v
        np.save(out_dir / f"wide_{f}.npy", m)
        print("field", f, "n", len(v), flush=True)
        del m, v
    return {"dates": int(nd), "symbols": int(ns), "rows_in": int(rows_in), "rows_kept": int(rows_kept)}
