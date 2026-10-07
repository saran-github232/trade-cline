"""CART-style binary decision tree for binary classification.

Split search is **histogram based**: every feature is pre-binned into at most
``max_bins`` quantile buckets, so evaluating a node costs ``O(n_features * n)``
instead of ``O(n_features * n log n)``.  This keeps pure-Python training fast
enough to bag dozens of trees while remaining exactly deterministic.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from typing import List, Optional, Sequence

from ..config import DEFAULT_SEED
from ..utils.stats import quantile as _quantile
from .base import BaseModel

__all__ = ["DecisionTreeModel"]


class _TreeNode:
    """A single node of the tree (a leaf when ``feature is None``)."""

    __slots__ = ("feature", "threshold", "left", "right", "prob", "n")

    def __init__(self, prob: float, n: int) -> None:
        self.feature: Optional[int] = None
        self.threshold: float = 0.0
        self.left: Optional["_TreeNode"] = None
        self.right: Optional["_TreeNode"] = None
        self.prob: float = prob
        self.n: int = n


class DecisionTreeModel(BaseModel):
    """Binary CART decision tree.

    Parameters
    ----------
    max_depth:
        Maximum tree depth (root has depth 0).
    min_samples_split:
        Minimum node size required to attempt a split.
    min_samples_leaf:
        Minimum number of samples that must remain on each side of a split.
    criterion:
        ``"gini"`` or ``"entropy"`` impurity measure.
    max_features:
        ``None``/``"all"``, ``"sqrt"``, ``"log2"``, a float fraction or an int
        count of features considered at each split.
    max_bins:
        Number of histogram buckets used for split search.
    seed:
        Seed for the per-node feature subsampling RNG.
    """

    name = "decision_tree"
    _param_names = (
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "criterion",
        "max_features",
        "max_bins",
        "seed",
        "class_weight",
    )

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        max_features: Optional[object] = None,
        max_bins: int = 32,
        seed: int = DEFAULT_SEED,
        class_weight: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = str(criterion)
        self.max_features = max_features
        self.max_bins = int(max_bins)
        self.seed = int(seed)
        self.class_weight = class_weight
        self._root: Optional[_TreeNode] = None
        self._edges: List[List[float]] = []
        self._rng = random.Random(self.seed)

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "DecisionTreeModel":
        """Grow the tree recursively; returns ``self``."""
        Xi, yb, w = self._fit_prepare(
            X,
            y,
            feature_names=feature_names,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        # A fresh RNG per fit makes repeated fits on identical data identical.
        self._rng = random.Random(self.seed)
        if self._single_class:
            self._edges = []
            self._root = _TreeNode(float(self._constant_prob or 0.0), len(Xi))
            self._fitted = True
            return self
        self._edges = self._compute_edges(Xi)
        self._root = self._build(list(range(len(Xi))), Xi, yb, w, 0)
        total = sum(self._importance)
        if total > 0:
            self._importance = [v / total for v in self._importance]
        self._fitted = True
        return self

    # -- split search helpers ---------------------------------------------
    def _compute_edges(self, X: List[List[float]]) -> List[List[float]]:
        """Pre-bin every feature into at most ``max_bins`` quantile buckets.

        Returns, per feature, the strictly increasing internal edges.  A
        feature that is constant (or has fewer than two distinct values) gets
        no edges and therefore can never be chosen for a split.
        """
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

    def _impurity(self, pos_w: float, total_w: float) -> float:
        """Weighted Gini or entropy impurity for a two-class node."""
        if total_w <= 0.0:
            return 0.0
        p = pos_w / total_w
        if self.criterion == "entropy":
            out = 0.0
            if p > 0.0:
                out -= p * math.log2(p)
            if p < 1.0:
                out -= (1.0 - p) * math.log2(1.0 - p)
            return out
        return 2.0 * p * (1.0 - p)  # Gini = 1 - p^2 - (1-p)^2

    def _find_split(
        self,
        indices: List[int],
        X: List[List[float]],
        y: List[int],
        w: List[float],
    ) -> Optional[tuple]:
        """Find the best ``(feature, threshold)`` split using histograms.

        Also accumulates the impurity decrease into ``self._importance`` so
        importance is computed in a single pass over the tree.
        """
        n = len(indices)
        tw = 0.0
        tp = 0.0
        for i in indices:
            tw += w[i]
            if y[i] == 1:
                tp += w[i]
        if tw <= 0.0:
            return None
        parent = self._impurity(tp, tw)
        best_gain = 0.0
        best: Optional[tuple] = None
        for f in self._candidate_features():
            es = self._edges[f]
            if not es:
                continue
            n_bins = len(es) + 1
            hist_w = [0.0] * n_bins
            hist_p = [0.0] * n_bins
            hist_n = [0] * n_bins
            for i in indices:
                b = bisect_right(es, X[i][f])
                hist_w[b] += w[i]
                hist_n[b] += 1
                if y[i] == 1:
                    hist_p[b] += w[i]
            wl = 0.0
            pl = 0.0
            nl = 0
            for b in range(n_bins - 1):
                wl += hist_w[b]
                pl += hist_p[b]
                nl += hist_n[b]
                nr = n - nl
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                wr = tw - wl
                if wl <= 0.0 or wr <= 0.0:
                    continue
                imp_l = self._impurity(pl, wl)
                imp_r = self._impurity(tp - pl, wr)
                gain = parent - (wl / tw) * imp_l - (wr / tw) * imp_r
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (f, es[b])
        if best is not None:
            self._importance[best[0]] += best_gain * n
            return best
        return None


    def _build(
        self,
        indices: List[int],
        X: List[List[float]],
        y: List[int],
        w: List[float],
        depth: int,
    ) -> _TreeNode:
        """Recursively grow the tree from ``indices``."""
        n = len(indices)
        tw = 0.0
        tp = 0.0
        for i in indices:
            tw += w[i]
            if y[i] == 1:
                tp += w[i]
        prob = (tp / tw) if tw > 0.0 else 0.0
        node = _TreeNode(prob, n)
        # Stop conditions: depth, size, or a pure node (no impurity to remove).
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or n < 2 * self.min_samples_leaf
            or tp <= 0.0
            or tp >= tw
        ):
            return node
        best = self._find_split(indices, X, y, w)
        if best is None:
            return node
        f, thr = best
        left = [i for i in indices if X[i][f] < thr]
        right = [i for i in indices if X[i][f] >= thr]
        if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
            return node
        node.feature = f
        node.threshold = thr
        node.left = self._build(left, X, y, w, depth + 1)
        node.right = self._build(right, X, y, w, depth + 1)
        return node

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Return the leaf class-1 frequency for each row."""
        Xp = self._prepare_predict(X)
        if self._single_class and self._constant_prob is not None:
            return [float(self._constant_prob)] * len(Xp)
        out: List[float] = []
        for xi in Xp:
            node = self._root
            while node is not None and node.feature is not None:
                node = node.left if xi[node.feature] < node.threshold else node.right
            out.append(float(node.prob) if node is not None else 0.0)
        return out

