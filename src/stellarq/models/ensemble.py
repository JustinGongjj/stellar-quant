"""DoubleEnsemble 式集成 + 概率校准 + 严格 walk-forward。

吸收来源
--------
* **DoubleEnsemble（Qlib，arXiv 2010.01265）**：每轮子模型训练后，
  ① 按「当前集成损失」与「损失下降速率」给样本重加权（SR）；
  ② 按「打乱某特征后集成损失增量」做特征选择（FS）。
  本项目按 A 股截面场景改写：样本 = (日期, 股票)，损失以**截面 rank IC 的反向**衡量，
  重加权按「日期分组」而非逐样本，避免同一交易日的股票被拆散。
* **概率校准**：把模型打分映射为「次日/次周跑赢中位数的概率」，
  用可靠性曲线 + Brier 分数 + 期望校准误差（ECE）评估，校准器用 Platt（逻辑回归）
  在验证集上拟合，测试集只做评估——不参与训练与校准。
* **walk-forward**：训练 / 验证（校准 + 早停）/ 测试三段严格按时间切分，测试段模型冻结。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .gbdt import GBDTParams, HistGBDT


# ------------------------------------------------------------------ 校准
def platt_fit(score: np.ndarray, y: np.ndarray, iters: int = 300, lr: float = 0.1) -> tuple[float, float]:
    """一维逻辑回归（Platt scaling）：p = 1/(1+exp(-(a·x+b)))，梯度下降拟合。"""
    x = np.asarray(score, dtype=float)
    t = np.asarray(y, dtype=float)
    a, b = 1.0, 0.0
    n = max(len(x), 1)
    for _ in range(iters):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g_a = float(((p - t) * x).sum()) / n
        g_b = float((p - t).sum()) / n
        a -= lr * g_a
        b -= lr * g_b
    return float(a), float(b)


def platt_apply(score: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * np.asarray(score, dtype=float) + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "pred": float(p[m].mean()), "actual": float(y[m].mean())})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gap"] = df["actual"] - df["pred"]
    return df


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    rel = reliability(p, y, bins)
    if rel.empty:
        return float("nan")
    return float((rel["n"] / rel["n"].sum() * rel["gap"].abs()).sum())


# ------------------------------------------------------------------ 集成
@dataclass
class DEConfig:
    n_models: int = 4
    keep_features: float = 0.7        # 特征选择后保留比例
    sr_decay: float = 0.9             # 样本重加权的时间衰减
    gbdt: GBDTParams = field(default_factory=GBDTParams)


class DoubleEnsembleRegressor:
    """多轮 GBDT + 日期级样本重加权 + 置换式特征选择。

    与 Qlib 原版的差异（写在报告里，避免误读）：
    * 损失用截面 RankIC 的负数（−IC）而非 MSE，因为目标是排序；
    * 样本权重按「交易日」给，同一交易日所有股票共享权重（避免把时序结构打散）；
    * 特征选择用训练集上置换后的 −IC 下降量（Qlib 用 MSE 增量）。
    """

    def __init__(self, cfg: DEConfig | None = None):
        self.cfg = cfg or DEConfig()
        self.models: list[HistGBDT] = []
        self.feature_sets: list[list[str]] = []
        self.history: list[dict] = []

    @staticmethod
    def _ic(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> float:
        ics = []
        for d in np.unique(dates):
            m = dates == d
            if m.sum() < 10:
                continue
            a, b = pred[m], y[m]
            if a.std() == 0 or b.std() == 0:
                continue
            ra = a.argsort().argsort().astype(float)
            rb = b.argsort().argsort().astype(float)
            ics.append(np.corrcoef(ra, rb)[0, 1])
        return float(np.mean(ics)) if ics else 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray, cols: list[str],
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            dates_val: np.ndarray | None = None) -> "DoubleEnsembleRegressor":
        weights = np.ones(len(y))
        feats = list(range(len(cols)))
        pred_ens = np.zeros(len(y))
        for k in range(self.cfg.n_models):
            idx_feat = feats
            model = HistGBDT(self.cfg.gbdt)
            # HistGBDT 不直接吃 sample_weight：用重复抽样近似（权重分位 → 重复次数）
            if k > 0 and len(np.unique(weights)) > 1:
                wq = pd.qcut(pd.Series(weights).rank(method="first"), 5, labels=False)
                rep = (wq + 1).to_numpy()
                rows = np.repeat(np.arange(len(y)), rep)
            else:
                rows = np.arange(len(y))
            model.fit(X[rows][:, idx_feat], y[rows], X_val[:, idx_feat] if X_val is not None else None,
                      y_val,
                      val_metric=(lambda p, t: self._ic(p, t, dates_val)) if dates_val is not None else None)
            self.models.append(model)
            self.feature_sets.append([cols[i] for i in idx_feat])
            p = model.predict(X[:, idx_feat])
            pred_ens = (pred_ens * k + p) / (k + 1)
            ic = self._ic(pred_ens, y, dates)
            self.history.append({"round": k, "n_features": len(idx_feat), "train_ic": ic,
                                 "best_iter": int(model.best_iteration_)})
            # 样本重加权：按日期给权重，表现差的日期权重高（最多 1/权重下限=10x）
            per_date = {}
            for d in np.unique(dates):
                m = dates == d
                if m.sum() < 10:
                    continue
                a, b = pred_ens[m], y[m]
                if a.std() == 0 or b.std() == 0:
                    continue
                ra = a.argsort().argsort().astype(float)
                rb = b.argsort().argsort().astype(float)
                per_date[d] = np.corrcoef(ra, rb)[0, 1]
            if len(per_date) > 2:
                s = pd.Series(per_date)
                rank = s.rank(pct=True)                 # IC 低的日期 → 权重高
                w_map = 1.0 / (self.cfg.sr_decay ** k * (1.0 - rank + 0.1))
                weights = np.array([w_map.get(d, 1.0) for d in dates], dtype=float)
            # 特征选择：在训练集上置换每列，看 −IC 下降
            if k + 1 < self.cfg.n_models:
                base_ic = self._ic(pred_ens, y, dates)
                gains = []
                rng = np.random.default_rng(1234 + k)
                for j in idx_feat:
                    Xp = X[:, idx_feat].copy()
                    Xp[:, list(idx_feat).index(j)] = rng.permutation(Xp[:, list(idx_feat).index(j)])
                    pm = np.zeros(len(y))
                    for mm in self.models:
                        pm += mm.predict(Xp[:, :len(self.feature_sets[self.models.index(mm)])]) / len(self.models)
                    gains.append((j, base_ic - self._ic(pm, y, dates)))
                gains.sort(key=lambda t: -t[1])
                keep = max(int(len(idx_feat) * self.cfg.keep_features), 20)
                feats = sorted(j for j, _ in gains[:keep])
        return self

    def predict(self, X: np.ndarray, cols: list[str]) -> np.ndarray:
        out = np.zeros(len(X))
        for model, fs in zip(self.models, self.feature_sets):
            idx = [cols.index(c) for c in fs]
            out += model.predict(X[:, idx])
        return out / max(len(self.models), 1)


# ------------------------------------------------------------------ walk-forward + 校准
@dataclass
class WFSpec:
    train_days: int = 750
    valid_days: int = 120
    test_days: int = 60        # 与标签周期一致（非重叠），避免段间信息泄漏
    min_train_rows: int = 20000
    de: DEConfig = field(default_factory=DEConfig)


def walk_forward(frame: pd.DataFrame, feature_cols: list[str], label_col: str, spec: WFSpec) -> dict:
    """滚动训练 + Platt 校准，返回逐段预测、校准曲线与统计。

    ``frame``：date / symbol / label / 特征列。测试段每只股票的标签是「相对截面中位数是否跑赢」，
    因此概率的含义是「跑赢当期中位数的概率」。
    """
    df = frame.dropna(subset=[label_col]).copy()
    df = df.sort_values(["date", "symbol"])
    dates = np.sort(df["date"].unique())
    rounds, preds = [], []
    start = spec.train_days + spec.valid_days
    while start < len(dates):
        tr_d = dates[max(0, start - spec.valid_days - spec.train_days): start - spec.valid_days]
        va_d = dates[start - spec.valid_days: start]
        te_d = dates[start: start + spec.test_days]
        if len(te_d) == 0:
            break
        tr = df[df["date"].isin(tr_d)]
        va = df[df["date"].isin(va_d)]
        te = df[df["date"].isin(te_d)]
        if len(tr) < spec.min_train_rows:
            start += spec.test_days
            continue
        Xtr = tr[feature_cols].to_numpy(dtype=float)
        Xva = va[feature_cols].to_numpy(dtype=float)
        Xte = te[feature_cols].to_numpy(dtype=float)
        ytr = tr[label_col].to_numpy(dtype=float)
        yva = va[label_col].to_numpy(dtype=float)
        yte = te[label_col].to_numpy(dtype=float)
        dtr = tr["date"].to_numpy()
        dva = va["date"].to_numpy()
        dte = te["date"].to_numpy()
        de = DoubleEnsembleRegressor(spec.de)
        de.fit(Xtr, ytr, dtr, feature_cols, Xva, yva, dva)
        sv = de.predict(Xva, feature_cols)
        st = de.predict(Xte, feature_cols)
        # 校准目标：相对截面中位数跑赢（训练/验证/测试各自按当日中位数定 0/1）
        def rel_win(y, d):
            out = np.zeros(len(y))
            for dd in np.unique(d):
                m = d == dd
                out[m] = (y[m] > np.median(y[m])).astype(float)
            return out
        yv_win = rel_win(yva, dva)
        yt_win = rel_win(yte, dte)
        a, b = platt_fit(sv, yv_win)
        p_te = platt_apply(st, a, b)
        preds.append(pd.DataFrame({"date": te["date"].to_numpy(), "symbol": te["symbol"].to_numpy(),
                                   "score": st, "prob": p_te, "label": yte, "win": yt_win}))
        rounds.append({
            "test_start": str(pd.Timestamp(te_d[0]).date()), "test_end": str(pd.Timestamp(te_d[-1]).date()),
            "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
            "valid_ic": DoubleEnsembleRegressor._ic(sv, yva, dva),
            "test_ic": DoubleEnsembleRegressor._ic(st, yte, dte),
            "test_brier": brier(p_te, yt_win), "test_brier_uncal": brier(platt_apply(st, 1.0, 0.0), yt_win),
            "test_ece": ece(p_te, yt_win),
            "platt": [a, b], "rounds": de.history,
        })
        start += spec.test_days
    out = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return {"predictions": out, "rounds": rounds}


def ic_by_date(pred: pd.DataFrame) -> pd.Series:
    out = {}
    for d, g in pred.groupby("date"):
        if len(g) < 10:
            continue
        a, b = g["score"].to_numpy(), g["label"].to_numpy()
        if a.std() == 0 or b.std() == 0:
            continue
        ra = a.argsort().argsort().astype(float)
        rb = b.argsort().argsort().astype(float)
        out[d] = float(np.corrcoef(ra, rb)[0, 1])
    return pd.Series(out).sort_index()
