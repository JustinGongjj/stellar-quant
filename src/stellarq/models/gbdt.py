"""LightGBM 的纯 Python 替代：直方图梯度提升树（回归 / 排序友好）。

为什么要自己写
--------------
本环境未安装 lightgbm，而下载外部 wheel 依赖不确定；同时核心需求（截面排序预测）
用「直方图分箱 + 二阶梯度 + 行列采样 + 提前停止」就能覆盖 95% 的场景。实现参照
LightGBM 的核心思想（Histogram-based split finding、leaf-wise 生长、λ 正则的叶子值求解），
并吸收 Qlib GBDT 封装里的两条工程约定：

* 训练/验证分离，按验证集 IC 早停（Qlib ``early_stopping_rounds``）；
* 输出为连续打分，评估用 RankIC 而非 MSE（Qlib SignalRecord + SigAnaRecord）。

接口与 sklearn 保持一致的子集（fit / predict / feature_importances_）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GBDTParams:
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_samples: int = 200
    lambda_l2: float = 10.0
    max_bins: int = 63
    subsample: float = 0.8
    colsample: float = 0.8
    early_stopping_rounds: int = 40
    loss: str = "l2"          # "l2" 或 "huber"
    huber_delta: float = 0.02
    random_state: int = 42


class HistGBDT:
    """极简直方图 GBDT。

    特征分箱在 ``fit`` 时按全样本分位数确定（in-sample 分箱；对时序预测不会引入未来信息，
    因为分箱只影响切分点的集合，不依赖标签）。训练时按 ``subsample`` 抽行、``colsample`` 抽列。
    """

    def __init__(self, params: GBDTParams | None = None):
        self.p = params or GBDTParams()
        self.bin_edges_: list[np.ndarray] = []
        self.trees_: list[dict] = []
        self.feature_importances_: np.ndarray | None = None
        self.best_iteration_: int = 0
        self.history_: list[dict] = []

    # ------------------------------------------------------------------ 分箱
    def _fit_bins(self, X: np.ndarray) -> None:
        qs = np.linspace(0, 1, self.p.max_bins + 1)[1:-1]
        self.bin_edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]

    def _bin(self, X: np.ndarray) -> np.ndarray:
        out = np.empty_like(X, dtype=np.int32)
        for j, edges in enumerate(self.bin_edges_):
            out[:, j] = np.searchsorted(edges, X[:, j], side="right")
        return out

    # ------------------------------------------------------------------ 训练
    def _leaf(self, grad: np.ndarray, hess: np.ndarray) -> float:
        return float(-grad.sum() / (hess.sum() + self.p.lambda_l2))

    def _build_tree(self, Xb: np.ndarray, grad: np.ndarray, hess: np.ndarray,
                    rows: np.ndarray, feats: np.ndarray) -> dict:
        max_depth = self.p.max_depth
        nb = self.p.max_bins

        def build(idx: np.ndarray, depth: int) -> dict:
            w = hess[idx].sum() + self.p.lambda_l2
            leaf = {"leaf": float(-grad[idx].sum() / w)}
            if depth >= max_depth or len(idx) < 2 * self.p.min_child_samples:
                return leaf
            best = None
            g_all = grad[idx].sum()
            h_all = hess[idx].sum()
            for f in feats:
                b = Xb[idx, f]
                gh = np.bincount(b, weights=grad[idx], minlength=nb)
                hh = np.bincount(b, weights=hess[idx], minlength=nb)
                cn = np.bincount(b, minlength=nb)
                g_l = np.cumsum(gh)[:-1]
                h_l = np.cumsum(hh)[:-1]
                c_l = np.cumsum(cn)[:-1]
                c_r = len(idx) - c_l
                ok = (c_l >= self.p.min_child_samples) & (c_r >= self.p.min_child_samples)
                if not ok.any():
                    continue
                g_r = g_all - g_l
                h_r = h_all - h_l
                gain = g_l**2 / (h_l + self.p.lambda_l2) + g_r**2 / (h_r + self.p.lambda_l2) \
                    - g_all**2 / (h_all + self.p.lambda_l2)
                gain = np.where(ok, gain, -np.inf)
                k = int(np.argmax(gain))
                if best is None or gain[k] > best[0]:
                    best = (float(gain[k]), int(f), int(k))
            if best is None or best[0] <= 0:
                return leaf
            _, f, k = best
            left = idx[Xb[idx, f] <= k]
            right = idx[Xb[idx, f] > k]
            if len(left) == 0 or len(right) == 0:
                return leaf
            return {"feat": f, "thr": k, "gain": best[0],
                    "left": build(left, depth + 1), "right": build(right, depth + 1)}

        return build(np.asarray(rows), 0)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            val_metric=None) -> "HistGBDT":
        """训练。``val_metric(pred_val, y_val) -> float`` 用于早停（越大越好，如 RankIC）。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._fit_bins(X)
        Xb = self._bin(X)
        n, m = X.shape
        base = float(np.mean(y))
        pred = np.full(n, base)
        self.trees_ = []
        self.feature_importances_ = np.zeros(m)
        rng = np.random.default_rng(self.p.random_state)
        best_score = -np.inf
        best_trees = []
        pred_val = np.full(len(y_val), base) if y_val is not None else None
        for it in range(self.p.n_estimators):
            if self.p.loss == "huber":
                r = y - pred
                d = self.p.huber_delta
                grad = -np.clip(r, -d, d)
                hess = np.ones_like(r)
            else:
                grad = pred - y
                hess = np.ones_like(grad)
            rows = rng.choice(n, size=max(int(n * self.p.subsample), self.p.min_child_samples * 2), replace=False)
            feats = rng.choice(m, size=max(int(m * self.p.colsample), 1), replace=False)
            tree = self._build_tree(Xb, grad, hess, np.sort(rows), np.sort(feats))
            self._accumulate_importance(tree, m)
            self.trees_.append(tree)
            pred += self.p.learning_rate * self._predict_tree_binned(Xb, tree)
            self.history_.append({"iter": it})
            if y_val is not None and val_metric is not None:
                if it == 0:
                    pred_val = np.full(len(y_val), base)
                else:
                    pred_val = pred_val + self.p.learning_rate * self._predict_tree(
                        self._bin(X_val), tree)
                score = float(val_metric(pred_val, y_val))
                self.history_[-1]["val_score"] = score
                if score > best_score + 1e-8:
                    best_score = score
                    best_trees = list(self.trees_)
                    self.best_iteration_ = it + 1
                elif it - self.best_iteration_ >= self.p.early_stopping_rounds:
                    break
        if best_trees and self.p.early_stopping_rounds:
            self.trees_ = best_trees
        self.base_score_ = base
        return self

    def _accumulate_importance(self, node: dict, m: int) -> None:
        if "leaf" in node:
            return
        self.feature_importances_[node["feat"]] += node["gain"]
        self._accumulate_importance(node["left"], m)
        self._accumulate_importance(node["right"], m)

    # ------------------------------------------------------------------ 预测
    def _predict_tree(self, Xb: np.ndarray, tree: dict) -> np.ndarray:
        out = np.empty(len(Xb))
        stack = [(tree, np.arange(len(Xb)))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if "leaf" in node:
                out[idx] = node["leaf"]
                continue
            go_left = Xb[idx, node["feat"]] <= node["thr"]
            stack.append((node["left"], idx[go_left]))
            stack.append((node["right"], idx[~go_left]))
        return out

    def _predict_tree_binned(self, Xb: np.ndarray, tree: dict) -> np.ndarray:
        return self._predict_tree(Xb, tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("模型尚未训练")
        X = np.asarray(X, dtype=float)
        Xb = self._bin(X)
        out = np.full(len(X), self.base_score_)
        for tree in self.trees_:
            out += self.p.learning_rate * self._predict_tree(Xb, tree)
        return out


def rank_ic_metric(pred: np.ndarray, y: np.ndarray) -> float:
    """皮尔逊相关（在排序分与收益上），用于早停；不依赖 scipy。"""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(y, dtype=float)
    if len(p) < 10 or np.std(p) == 0 or np.std(t) == 0:
        return 0.0
    pr = p.argsort().argsort().astype(float)
    tr = t.argsort().argsort().astype(float)
    return float(np.corrcoef(pr, tr)[0, 1])
