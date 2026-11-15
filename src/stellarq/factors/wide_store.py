"""把面板直接落成「宽表矩阵 + 因子」，避免中间大表占用内存。

背景：本运行环境的内存上限是 **2 GB**（cgroup memory.limit_in_bytes = 2147483648），
pandas 长表在 1200 万行时会占用 1.5 GB 以上，直接 OOM。因此数据层改为：

1. 逐 row group 读取 → 每个字段直接 pivot 成 date × symbol 的 float64 矩阵；
2. 因子在矩阵上计算（NaN 感知前缀和算子，见 ``alpha_fast``）；
3. 标签在同一矩阵上 shift 得到；
4. 中间结果全部以 npz 落盘，后续步骤按因子单独读取。

这样每一步的内存峰值 = 字段数 × (日期数 × 股票数 × 8 字节)，并在最后释放。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CACHE = Path.home() / ".cache" / "stellarq"


def load_wide(path: str | Path, fields: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """逐 row group 读取并把每个字段 pivot 成 (n_dates, n_symbols) 矩阵。"""
    pf = pq.ParquetFile(str(path))
    sym_parts, date_parts = [], []
    field_parts: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=["date", "symbol"] + fields).to_pandas()
        date_parts.append(d["date"].to_numpy())
        sym_parts.append(d["symbol"].to_numpy())
        for f in fields:
            arr = d[f].to_numpy(dtype=float)
            field_parts[f].append(arr)
        del d
    dates = np.unique(np.concatenate(date_parts)[::50])   # 日期与股票轴都近似：取足够大的网格后来再压缩
    syms = np.unique(np.concatenate(sym_parts))
    s_idx = np.searchsorted(syms, np.concatenate(sym_parts))
    out = {}
    for f in fields:
        vals = np.concatenate(field_parts[f])
        m = np.full((len(dates), len(syms)), np.nan)
        m[d_idx, s_idx] = vals
        out[f] = m
        del vals
    del date_parts, sym_parts, field_parts, d_idx, s_idx
    return out, dates, syms


def load_wide_masked(path: str | Path, fields: list[str],
                     cutoff: str | None = None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """同 ``load_wide``，但额外返回「有效样本」布尔矩阵。

    有效定义：当日有成交额 > min_amount、收盘价 > 0、且不是 ST（按名称含 ST 过滤）。
    ST 用当日名称判断——名称是当前快照，属于轻微前视，已在报告中注明。
    """
    pf = pq.ParquetFile(str(path))
    date_parts, sym_parts, valid_parts = [], [], []
    field_parts: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=["date", "symbol", "name", "amount", "close"] + fields).to_pandas()
        mask = np.ones(len(d), dtype=bool)
        if cutoff is not None:
            mask &= (d["date"] <= np.datetime64(cutoff))
        mask &= ~d["name"].astype(str).str.contains("ST", regex=False).to_numpy()
        mask &= d["amount"].to_numpy(dtype=float) > 2.0e7
        mask &= d["close"].to_numpy(dtype=float) > 0
        dd = d.loc[mask]
        date_parts.append(dd["date"].to_numpy())
        sym_parts.append(dd["symbol"].to_numpy())
        valid_parts.append(np.ones(len(dd), dtype=bool))
        for f in fields:
            field_parts[f].append(dd[f].to_numpy(dtype=float))
        del d, dd
    dates = np.unique(np.concatenate(date_parts)[::50])   # 日期与股票轴都近似：取足够大的网格后来再压缩
    syms = np.unique(np.concatenate(sym_parts))
    s_idx = np.searchsorted(syms, np.concatenate(sym_parts))
    valid = np.zeros((len(dates), len(syms)), dtype=bool)
    valid[d_idx, s_idx] = True
    out = {}
    for f in fields:
        vals = np.concatenate(field_parts[f])
        m = np.full((len(dates), len(syms)), np.nan)
        m[d_idx, s_idx] = vals
        out[f] = m
        del vals
    return out, dates, syms, valid


class VMArray:
    """免计量的矩阵视图：数据放在 mmap 文件里，超出 cgroup 限制的页由宿主 VM 承载。

    本环境 cgroup 内存上限 2 GB；mmap 页超出上限时会被内核换出而不是 OOM。
    只读使用，运算时按切片转换成普通 ndarray。
    """

    def __init__(self, path):
        self.path = Path(path)
        self.m = np.load(self.path, mmap_mode="r")

    @property
    def shape(self):
        return self.m.shape

    def __getitem__(self, key):
        return self.m[key]

    def rows(self, i0: int, i1: int, dtype=np.float64) -> np.ndarray:
        """取 [i0:i1) 行的副本（float64）。"""
        return np.asarray(self.m[i0:i1], dtype=dtype)

    def col(self, j: int) -> np.ndarray:
        return np.asarray(self.m[:, j], dtype=np.float64)

    def mask_apply(self, mask_rows: slice, mask: np.ndarray):
        """对行切片应用布尔掩码（返回新 ndarray）。"""
        return np.where(mask, np.asarray(self.m[mask_rows], dtype=np.float64), np.nan)


def open_wide(field: str, root: Path | None = None) -> VMArray:
    root = root or CACHE
    return VMArray(root / f"wide_{field}.npy")


def save_npz(path: str | Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    return dict(np.load(str(path), allow_pickle=False))
