"""修复 v4_X.npy 里全零的 LN_CAP 列。

问题
----
`v4_X.npy` 19 列中，列 1（`LN_CAP`，市值对数）整列 std = 0。
经核对：当前 `model_v4.py` 的计算逻辑**没问题**（log1p(float_mktcap) 正常产出约 25–26 的值），
说明缓存是旧版脚本产物。

`LN_CAP` 是 A 股最强风格因子之一（小市值效应），缺失等于模型完全没有市值暴露。

方案
----
不重跑全量特征（要遍历 3500 日 × 5000 只，代价大），而是：
缓存里已有 `v4_d`（日期索引）和 `v4_s`（股票索引），
可以直接从 `wide_float_mktcap.npy` 按索引重建这一列。

安全措施
--------
· 先备份原文件
· 重建后校验：非零比例、量级、与已知市值分布一致
· 若校验不过，回滚
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "stellarq"
R = CACHE / "wide2"
LN_CAP_COL = 1


def main(dry_run: bool = False):
    X = np.load(CACHE / "v4_X.npy")
    d = np.load(CACHE / "v4_d.npy")
    s = np.load(CACHE / "v4_s.npy")
    cap = np.load(R / "wide_float_mktcap.npy", mmap_mode="r")

    print(f"X {X.shape} · d {d.shape} · s {s.shape} · cap {cap.shape}")

    old = X[:, LN_CAP_COL]
    print(f"修复前 列{LN_CAP_COL}: std={np.std(old):.6f}  非零 {int(np.sum(old != 0))}/{len(old)}")

    # v4_d / v4_s 存的是**面板索引**（不是日期字符串）
    print(f"  d 范围 [{d.min()}, {d.max()}] / 面板 {cap.shape[0]} 日")
    print(f"  s 范围 [{s.min()}, {s.max()}] / 面板 {cap.shape[1]} 只")

    if d.max() >= cap.shape[0] or s.max() >= cap.shape[1]:
        print("✗ 索引越界，v4_d/v4_s 不是面板索引，方案不适用")
        return False

    new = np.log1p(np.asarray(cap[d, s], dtype=np.float64)).astype(np.float32)
    fin = np.isfinite(new)
    print(f"\n重建 列{LN_CAP_COL}: 有效 {int(fin.sum())}/{len(new)} "
          f"({fin.mean()*100:.1f}%)  std={np.std(new[fin]):.4f}  "
          f"范围 [{np.min(new[fin]):.2f}, {np.max(new[fin]):.2f}]")

    # 校验：市值对数的合理区间 15–32（约 3 亿 ~ 8 万亿）
    if not (fin.mean() > 0.5 and 15 < np.median(new[fin]) < 32):
        print("✗ 校验不通过（有效比例或量级异常），不写回")
        return False

    if dry_run:
        print("\n[dry-run] 未写回")
        return True

    bak = CACHE / "v4_X.npy.bak"
    if not bak.exists():
        shutil.copy2(CACHE / "v4_X.npy", bak)
        print(f"\n已备份 → {bak.name}")

    X[:, LN_CAP_COL] = new
    np.save(CACHE / "v4_X.npy", X)
    print("✓ 已写回 v4_X.npy")

    # 复验
    Y = np.load(CACHE / "v4_X.npy")
    col = Y[:, LN_CAP_COL]
    fin2 = np.isfinite(col)
    print(f"复验: std={np.std(col[fin2]):.4f}  有效 {int(fin2.sum())}/{len(col)}")
    return True


if __name__ == "__main__":
    ok = main(dry_run="--dry-run" in sys.argv)
    print("\n结果:", "成功" if ok else "失败")