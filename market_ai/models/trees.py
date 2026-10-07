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
