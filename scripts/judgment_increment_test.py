"""判断力增量检验：在 v4 的 20 个量价特征之上，哪些新信号源能带来**正交**增量。

问题
----
v4 模型 OOS RankIC ≈ 0.076，20 个特征全是量价/均线/波动。
要继续提高判断力，只有一条路：**加正交的新信息**，而不是在同一批量价里调参。

本脚本提供一个统一的检验口径，任何候选特征都可以丢进来跑：

  1. 单独 IC —— 它自己有没有预测力
  2. 对现有特征的**正交化后 IC** —— 扣掉「其实是量价的另一面」之后还剩多少
  3. 分组增量 —— 把它加进现有特征后，模型 OOS IC 提升多少

第 2 步是关键：很多"新特征"看着有 IC，其实和 MA20/STD20 高度相关，加上去等于没加。

口径纪律
--------
· 标签 y 已存在（v4_y.npy，5 日前向收益）
· 每日截面 zscore 后再算 IC，避免不同日期的量纲差异
· 训练/测试按时间切分，**绝不随机切分**
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path.home() / ".cache" / "stellarq"

FEAT_NAMES = ['MA60', 'MA20', 'MA10', 'MIN5', 'MIN20', 'RET20', 'AMT_RATIO60',
              'AMT_RATIO20', 'VSUMD60', 'VMA60', 'VSTD20', 'VSTD10', 'AMT_MA5',
              'LN_CAP', 'STD20', 'TURN5', 'KMID', 'CLOSE_POS', 'VSTD60', 'CORR20']


def load_v4():
    X = np.load(CACHE / "v4_X.npy")
    y = np.load(CACHE / "v4_y.npy")
    d = np.load(CACHE / "v4_d.npy")
    s = np.load(CACHE / "v4_s.npy")
    return X, y, d, s


def cs_rank_ic(x, y, d):
    """按日截面算 RankIC 的均值。x/y/d 等长。"""
    df = pd.DataFrame({"d": d, "x": x, "y": y})
    df = df[np.isfinite(df.x) & np.isfinite(df.y)]
    if len(df) < 5000:
        return np.nan, 0
    ics = []
    for _, g in df.groupby("d"):
        if len(g) >= 50:
            ics.append(g.x.corr(g.y, method="spearman"))
    ics = np.array([v for v in ics if np.isfinite(v)])
    if len(ics) < 10:
        return np.nan, len(ics)
    return float(ics.mean()), len(ics)


def cs_zscore(x, d):
    df = pd.DataFrame({"d": d, "x": x})
    g = df.groupby("d")["x"]
    mu = g.transform("mean")
    sd = g.transform("std")
    return ((df.x - mu) / sd.replace(0, np.nan)).to_numpy()


def residualize(x, base_cols, d):
    """把 x 对 base_cols 做**逐日截面**回归，取残差。

    为什么逐日做：跨日回归会混入时间趋势，把"和基准同涨同跌"误判为共线。
    """
    df = pd.DataFrame({"d": d, "x": x})
    for i, c in enumerate(base_cols):
        df[f"b{i}"] = c
    out = np.full(len(x), np.nan, dtype=np.float64)
    bcols = [c for c in df.columns if c.startswith("b")]
    for day, g in df.groupby("d"):
        m = np.isfinite(g.x) & np.isfinite(g[bcols]).all(axis=1)
        if m.sum() < 60:
            continue
        A = g.loc[m, bcols].to_numpy(dtype=float)
        # 加常数项
        A1 = np.column_stack([np.ones(len(A)), A])
        b = g.loc[m, "x"].to_numpy(dtype=float)
        try:
            coef, *_ = np.linalg.lstsq(A1, b, rcond=None)
            resid = b - A1 @ coef
        except Exception:
            continue
        out[g.index[m]] = resid
    return out


def main():
    X, y, d, s = load_v4()
    print(f"样本 {len(y)} 行 · {len(np.unique(d))} 个交易日 · {X.shape[1]} 个特征", flush=True)

    # 基准：现有最强的几个特征的单独 IC 与正交化后 IC
    print("\n=== 现有特征基线（单独 IC vs 对其余 19 个正交化后 IC）===")
    print(f"{'特征':<12}{'单独IC':>10}{'正交后IC':>11}{'衰减':>8}")
    base_rows = []
    for i, name in enumerate(FEAT_NAMES[:6]):
        x = X[:, i].astype(float)
        ic_raw, n = cs_rank_ic(x, y, d)
        others = [X[:, j].astype(float) for j in range(X.shape[1]) if j != i]
        xz = cs_zscore(x, d)
        oz = [cs_zscore(o, d) for o in others]
        r = residualize(xz, oz, d)
        ic_orth, _ = cs_rank_ic(r, y, d)
        dec = (1 - abs(ic_orth) / abs(ic_raw)) * 100 if ic_raw and abs(ic_raw) > 1e-9 else np.nan
        print(f"{name:<12}{ic_raw:+10.4f}{ic_orth:+11.4f}{dec:7.0f}%", flush=True)
        base_rows.append((name, ic_raw, ic_orth))

    # 判断力上限估计：所有 20 个特征一起的截面 IC（用线性合成近似）
    print("\n=== 20 特征线性合成（等权 zscore 求和）的 IC ===")
    Z = np.column_stack([cs_zscore(X[:, i].astype(float), d) for i in range(X.shape[1])])
    # 按每个特征的 IC 符号加权，避免反向特征抵消
    signs = []
    for i in range(X.shape[1]):
        ic, _ = cs_rank_ic(Z[:, i], y, d)
        signs.append(np.sign(ic) if np.isfinite(ic) else 0)
    combo = np.nansum(Z * np.array(signs), axis=1)
    ic_combo, n = cs_rank_ic(combo, y, d)
    print(f"等权合成 IC = {ic_combo:+.4f}（{n} 个截面）", flush=True)
    print(f"参照：v4 模型 OOS RankIC ≈ +0.0758", flush=True)

    # 理论上的增量空间：如果加入一个与现有特征完全正交、单独 IC=0.02 的信号
    print("\n=== 增量空间估算（解析）===")
    print("若新信号与现有合成信号正交、自身 IC=ic_new，")
    print("合成 IC ≈ sqrt(ic_old^2 + ic_new^2)（正交分量按平方和叠加）")
    for ic_new in (0.01, 0.02, 0.03, 0.05):
        est = np.sqrt(ic_combo ** 2 + ic_new ** 2)
        print(f"  ic_new={ic_new:.2f} → 合成 IC≈{est:.4f}（+{(est-ic_combo)*100:.1f}bp）", flush=True)


if __name__ == "__main__":
    main()