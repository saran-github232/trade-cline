"""Gradient boosting on the logistic loss with shallow regression trees.

Each boosting round fits a small regression tree to the negative gradient
(residuals) of the logistic loss using a second-order (Newton) leaf value, then
adds a shrunk contribution to the additive score.  An internal, time-ordered
validation split drives early stopping.  Everything is seeded and sequential,
so identical inputs and seeds produce identical predictions.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from typing import List, Optional, Sequence, Tuple

from ..config import DEFAULT_SEED
from ..utils.stats import log_loss as _log_loss
from ..utils.stats import quantile as _quantile
from .base import BaseModel, sigmoid

__all__ = ["GradientBoostingModel"]


class _RegNode:
    """A node of the internal regression tree (leaf when ``feature is None``)."""

    __slots__ = ("value", "feature", "threshold", "left", "right")

    def __init__(self, value: float) -> None:
        self.value = value
        self.feature: Optional[int] = None
        self.threshold: float = 0.0
        self.left: Optional["_RegNode"] = None
        self.right: Optional["_RegNode"] = None


class _RegressionTree:
    """Shallow regression tree fitted to gradients ``g`` and hessians ``h``.

    The leaf value is the Newton step ``sum(g) / (sum(h) + l2)`` and the split
    gain follows the XGBoost-style formula
    ``0.5 * (GL^2/(HL+l2) + GR^2/(HR+l2) - G^2/(H+l2))``.  Leaf values are
    clipped to ``[-max_value, max_value]`` to keep early rounds stable.
    """

    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        l2: float,
        max_features: Optional[object],
        seed: int,
        max_value: float = 5.0,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2 = float(l2)
        self.max_features = max_features
        self.seed = int(seed)
        self.max_value = float(max_value)
        self._rng = random.Random(self.seed)
        self._n_features = 0
        self._edges: List[List[float]] = []
        self._importance: List[float] = []
        self._root: Optional[_RegNode] = None

    def fit(
        self,
        X: List[List[float]],
        g: List[float],
        h: List[float],
        edges: List[List[float]],
    ) -> "_RegressionTree":
        """Grow the tree to fit the current gradients/hessians."""
        self._n_features = len(X[0]) if X else 0
        self._importance = [0.0] * self._n_features
        self._edges = edges
        self._rng = random.Random(self.seed)
        self._root = self._build(list(range(len(X))), X, g, h, 0)
        total = sum(self._importance)
        if total > 0:
            self._importance = [v / total for v in self._importance]
        return self

    def _candidate_features(self) -> List[int]:
        """Return the feature indices considered at the current node."""
        d = self._n_features
        mf = self.max_features
        if mf is None or (isinstance(mf, str) and mf == "all"):
            return list(range(d))
        if mf == "sqrt":
            k = max(1, int(math.sqrt(d)))
        elif mf == "log2":
            k = max(1, int(math.log2(d))) if d > 1 else 1
        elif isinstance(mf, float):
            k = max(1, int(round(mf * d)))
        else:
            k = max(1, min(d, int(mf)))
        if k >= d:
            return list(range(d))
        return self._rng.sample(range(d), k)

    def _leaf_value(self, g_sum: float, h_sum: float) -> float:
        """Newton leaf value, clipped to ``[-max_value, max_value]``."""
        value = g_sum / (h_sum + self.l2)
        if value > self.max_value:
            return self.max_value
        if value < -self.max_value:
            return -self.max_value
        return value

    def _build(
        self,
        idx: List[int],
        X: List[List[float]],
        g: List[float],
        h: List[float],
        depth: int,
    ) -> _RegNode:
        """Recursively grow the regression tree from ``idx``."""
        g_sum = sum(g[i] for i in idx)
        h_sum = sum(h[i] for i in idx)
        node = _RegNode(self._leaf_value(g_sum, h_sum))
        n = len(idx)
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return node
        best = self._find_split(idx, X, g, h, g_sum, h_sum, n)
        if best is None:
            return node
        f, thr, gain = best
        left = [i for i in idx if X[i][f] < thr]
        right = [i for i in idx if X[i][f] >= thr]
        if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
            return node
        self._importance[f] += gain
        node.feature = f
        node.threshold = thr
        node.left = self._build(left, X, g, h, depth + 1)
        node.right = self._build(right, X, g, h, depth + 1)
        return node

    def _find_split(
        self,
        idx: List[int],
        X: List[List[float]],
        g: List[float],
        h: List[float],
        g_sum: float,
        h_sum: float,
        n: int,
    ) -> Optional[Tuple[int, float, float]]:
        """Find the split maximising the second-order gain."""
        parent = g_sum * g_sum / (h_sum + self.l2)
        best_gain = 0.0
        best: Optional[Tuple[int, float, float]] = None
        for f in self._candidate_features():
            es = self._edges[f]
            if not es:
                continue
            n_bins = len(es) + 1
            hg = [0.0] * n_bins
            hh = [0.0] * n_bins
            hn = [0] * n_bins
            for i in idx:
                b = bisect_right(es, X[i][f])
                hg[b] += g[i]
                hh[b] += h[i]
                hn[b] += 1
            gl = 0.0
            hl = 0.0
            nl = 0
            for b in range(n_bins - 1):
                gl += hg[b]
                hl += hh[b]
                nl += hn[b]
                nr = n - nl
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gr = g_sum - gl
                hr = h_sum - hl
                gain = 0.5 * (
                    gl * gl / (hl + self.l2)
                    + gr * gr / (hr + self.l2)
                    - parent
                )
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (f, es[b], gain)
        return best

    def predict(self, X: List[List[float]]) -> List[float]:
        """Return the leaf value reached by each row."""
        out: List[float] = []
        for xi in X:
            node = self._root
            while node is not None and node.feature is not None:
                node = node.left if xi[node.feature] < node.threshold else node.right
            out.append(node.value if node is not None else 0.0)
        return out



class GradientBoostingModel(BaseModel):
    """Gradient-boosted logistic regression with shallow trees.

    Parameters
    ----------
    n_estimators:
        Maximum number of boosting rounds.
    learning_rate:
        Shrinkage applied to each tree's contribution.
    max_depth:
        Depth of each regression tree (shallow, e.g. 2-3).
    min_samples_leaf:
        Minimum leaf size in each regression tree.
    subsample:
        Row fraction used per round (``< 1`` introduces stochasticity).
    l2:
        L2 penalty inside the leaf value / split gain (stabilises Newton step).
    max_bins:
        Histogram buckets used for split search.
    validation_fraction:
        Fraction of the *latest* rows held out for early stopping.
    patience:
        Rounds without validation improvement before stopping.
    seed:
        Seed for subsampling (and tree seeds).
    class_weight:
        Optional ``"balanced"`` class weighting.
    """

    name = "gradient_boosting"
    _param_names = (
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "l2",
        "max_bins",
        "validation_fraction",
        "patience",
        "seed",
        "class_weight",
    )

    def __init__(
        self,
        n_estimators: int = 60,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        l2: float = 1.0,
        max_bins: int = 32,
        validation_fraction: float = 0.1,
        patience: int = 10,
        seed: int = DEFAULT_SEED,
        class_weight: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.l2 = float(l2)
        self.max_bins = int(max_bins)
        self.validation_fraction = float(validation_fraction)
        self.patience = int(patience)
        self.seed = int(seed)
        self.class_weight = class_weight
        self._base: float = 0.0
        self._trees: List[_RegressionTree] = []
        self._edges: List[List[float]] = []

    def _compute_edges(self, X: List[List[float]]) -> List[List[float]]:
        """Pre-bin features into quantile buckets shared by every round."""
        b = max(2, self.max_bins)
        edges: List[List[float]] = []
        for j in range(self._n_features):
            col = sorted(r[j] for r in X if math.isfinite(r[j]))
            if len(col) < 2 or col[0] == col[-1]:
                edges.append([])
                continue
            raw = []
            for k in range(1, b):
                q = _quantile(col, k / b)
                if math.isfinite(q):
                    raw.append(q)
            deduped: List[float] = []
            for e in raw:
                if not deduped or e > deduped[-1]:
                    deduped.append(e)
            edges.append(deduped)
        return edges

    def _split_validation(
        self, X: List[List[float]], y: List[int], w: List[float]
    ) -> Tuple[List[List[float]], List[int], List[float], List[List[float]], List[int]]:
        """Time-ordered train/validation split for early stopping.

        Uses the *latest* rows as validation so the split respects causality.
        Returns the validation lists empty when early stopping is disabled or
        there is too little data.
        """
        n = len(X)
        frac = self.validation_fraction
        if frac <= 0.0 or n < 20:
            return X, y, w, [], []
        cut = int(n * (1.0 - frac))
        cut = max(1, min(n - 1, cut))
        return X[:cut], y[:cut], w[:cut], X[cut:], y[cut:]


    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "GradientBoostingModel":
        """Run the boosting loop with early stopping; returns ``self``."""
        Xi, yb, w = self._fit_prepare(
            X,
            y,
            feature_names=feature_names,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        self._trees = []
        if self._single_class:
            self._base = 0.0
            self._fitted = True
            return self

        total_w = sum(w) or 1.0
        p0 = sum(w[i] * yb[i] for i in range(len(yb))) / total_w
        p0 = min(max(p0, 1e-6), 1.0 - 1e-6)
        self._base = math.log(p0 / (1.0 - p0))
        self._edges = self._compute_edges(Xi)

        Xtr, ytr, _wtr, Xval, yval = self._split_validation(Xi, yb, w)
        ntr = len(Xtr)
        f_tr = [self._base] * ntr
        f_val = [self._base] * len(Xval)
        rng = random.Random(self.seed)
        best_val = float("inf")
        best_iter = 0
        no_improve = 0

        for m in range(self.n_estimators):
            p = [sigmoid(f) for f in f_tr]
            g = [ytr[i] - p[i] for i in range(ntr)]
            h = [max(p[i] * (1.0 - p[i]), 1e-12) for i in range(ntr)]
            if self.subsample < 1.0:
                k = max(1, int(ntr * self.subsample))
                sel = rng.sample(range(ntr), k)
            else:
                sel = list(range(ntr))
            Xs = [Xtr[i] for i in sel]
            gs = [g[i] for i in sel]
            hs = [h[i] for i in sel]
            tree = _RegressionTree(
                self.max_depth,
                self.min_samples_leaf,
                self.l2,
                None,
                self.seed + 1000 * m + 1,
            )
            tree.fit(Xs, gs, hs, self._edges)
            self._trees.append(tree)

            upd = tree.predict(Xtr)
            for i in range(ntr):
                f_tr[i] += self.learning_rate * upd[i]

            if Xval:
                upv = tree.predict(Xval)
                for i in range(len(Xval)):
                    f_val[i] += self.learning_rate * upv[i]
                val_loss = _log_loss([sigmoid(f) for f in f_val], yval)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_iter = len(self._trees)
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

        # Revert to the best-scoring round when early stopping triggered.
        if Xval and 0 < best_iter < len(self._trees):
            self._trees = self._trees[:best_iter]

        self._importance = self._average_importance()
        self._fitted = True
        return self

    def _average_importance(self) -> List[float]:
        """Average per-feature importance across the retained trees."""
        if not self._trees:
            return [0.0] * self._n_features
        acc = [0.0] * self._n_features
        for tree in self._trees:
            for j in range(self._n_features):
                acc[j] += tree._importance[j]
        m = len(self._trees)
        return [v / m for v in acc]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Return ``sigmoid(base + sum(shrunk tree outputs))`` per row."""
        Xp = self._prepare_predict(X)
        if self._single_class and self._constant_prob is not None:
            return [float(self._constant_prob)] * len(Xp)
        scores = [self._base] * len(Xp)
        for tree in self._trees:
            upd = tree.predict(Xp)
            for i in range(len(Xp)):
                scores[i] += self.learning_rate * upd[i]
        return [sigmoid(s) for s in scores]

