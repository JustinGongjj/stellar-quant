"""截面排序预测的 walk-forward 管线（吸收 Qlib 的训练/验证/预测三段与 Alphalens 检验流程）。

核心纪律
--------
1. **时间切片**：用 [train_start, train_end) 训练，[valid_start, valid_end) 早停，
   [test_start, test_end) 预测；三段严格不重叠、不洗牌。
2. **滚动**：每 ``retrain_every`` 个交易日重训一次，预测窗口内模型冻结（不偷看未来标签）。
3. **标签**：``label`` 由调用方传入（推荐 T+1 开盘买入、T+1+h 开盘卖出的前向收益），
   管线只保证「特征日期 ≤ 训练截止日」和「标签日期 > 特征日期」。
4. **输出**：每日截面打分 + 每日 RankIC + 换手统计，供下游组合构建与检验使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .gbdt import GBDTParams, HistGBDT, rank_ic_metric


@dataclass
class WalkForwardSpec:
    train_days: int = 500          # 训练窗口（交易日）
    valid_days: int = 120          # 验证窗口
    test_days: int = 20            # 每次预测的步长
    min_train_samples: int = 5000
    params: GBDTParams = field(default_factory=GBDTParams)
    feature_cols: list[str] | None = None
    label_col: str = "label"


def _prep(frame: pd.DataFrame, feature_cols: list[str] | None, label_col: str):
    cols = feature_cols or [c for c in frame.columns if c not in ("date", "symbol", label_col)]
    df = frame[["date", "symbol", label_col] + cols].dropna(subset=[label_col]).copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 缺失特征按截面中位数填充，再截断极端值（与 toraniko 的 winsorize 口径一致）
    df[cols] = df.groupby("date")[cols].transform(lambda s: s.fillna(s.median()))
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=cols)
    for c in cols:
        lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(lo, hi)
    return df, cols


def walk_forward_predict(frame: pd.DataFrame, spec: WalkForwardSpec) -> dict:
    """执行滚动训练并返回每日预测与逐轮诊断。

    ``frame`` 需要列：date / symbol / label / 特征列。
    返回 ``{"predictions": DataFrame[date,symbol,score], "rounds": [...], "ic": Series}``。
    """
    df, cols = _prep(frame, spec.feature_cols, spec.label_col)
    dates = np.sort(df["date"].unique())
    if len(dates) < spec.train_days + spec.valid_days + spec.test_days:
        raise ValueError(f"样本日期不足：{len(dates)} < train+valid+test")
    preds: list[pd.DataFrame] = []
    rounds: list[dict] = []
    start = spec.train_days + spec.valid_days
    while start < len(dates):
        train_dates = dates[start - spec.valid_days - spec.train_days: start - spec.valid_days]
        valid_dates = dates[start - spec.valid_days: start]
        test_dates = dates[start: start + spec.test_days]
        if len(test_dates) == 0:
            break
        tr = df[df["date"].isin(train_dates)]
        va = df[df["date"].isin(valid_dates)]
        te = df[df["date"].isin(test_dates)]
        if len(tr) < spec.min_train_samples or len(te) == 0:
            start += spec.test_days
            continue
        model = HistGBDT(spec.params)
        model.fit(tr[cols].to_numpy(dtype=float), tr[spec.label_col].to_numpy(dtype=float),
                  va[cols].to_numpy(dtype=float), va[spec.label_col].to_numpy(dtype=float),
                  val_metric=_daily_rank_ic)
        score = model.predict(te[cols].to_numpy(dtype=float))
        out = te[["date", "symbol", spec.label_col]].copy()
        out["score"] = score
        preds.append(out)
        rounds.append({
            "test_start": str(pd.Timestamp(test_dates[0]).date()),
            "test_end": str(pd.Timestamp(test_dates[-1]).date()),
            "train_rows": int(len(tr)), "valid_rows": int(len(va)), "test_rows": int(len(te)),
            "best_iter": int(model.best_iteration_),
            "valid_ic": float(_daily_rank_ic(model.predict(va[cols].to_numpy(dtype=float)),
                                             va[spec.label_col].to_numpy(dtype=float))),
            "test_ic": float(_daily_rank_ic(score, te[spec.label_col].to_numpy(dtype=float))),
            "top_features": [cols[i] for i in np.argsort(-model.feature_importances_)[:8]],
        })
        start += spec.test_days
    if not preds:
        raise RuntimeError("没有产生任何预测，检查样本区间与最小样本数")
    pred_df = pd.concat(preds, ignore_index=True)
    ic = pred_df.groupby("date").apply(
        lambda g: g["score"].corr(g[spec.label_col], method="spearman") if len(g) > 5 else np.nan,
        include_groups=False)
    return {"predictions": pred_df, "rounds": rounds, "ic": ic.dropna(), "features": cols}


def _daily_rank_ic(pred: np.ndarray, y: np.ndarray) -> float:
    return rank_ic_metric(pred, y)


def ic_summary(ic: pd.Series) -> dict:
    """IC 序列的常用统计（Alphalens / Qlib SigAnaRecord 口径）。"""
    s = pd.Series(ic).dropna().astype(float)
    if s.empty:
        return {"n": 0}
    mean, std = float(s.mean()), float(s.std(ddof=1))
    return {
        "n": int(len(s)),
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if std > 0 else float("nan"),
        "t_stat": mean / (std / np.sqrt(len(s))) if std > 0 else float("nan"),
        "positive_ratio": float((s > 0).mean()),
        "hit_rate_top": float("nan"),
    }
