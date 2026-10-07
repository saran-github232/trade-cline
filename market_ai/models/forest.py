"""Random forest: bagged decision trees with feature subsampling and OOB.

Each tree is trained on an independent bootstrap sample drawn from a local
``random.Random(seed)`` so the whole ensemble is deterministic.  Out-of-bag
probabilities provide a free, leakage-free validation signal without touching a
held-out set.  ``n_jobs`` is accepted for API compatibility but the ensemble is
always built sequentially - that is precisely what guarantees determinism.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from ..config import DEFAULT_SEED
from .base import BaseModel
from .trees import DecisionTreeModel

__all__ = ["RandomForestModel"]


class RandomForestModel(BaseModel):
    """Bootstrap-aggregated ensemble of :class:`DecisionTreeModel`.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Per-tree depth cap.
    min_samples_split / min_samples_leaf:
        Per-tree split constraints.
    max_features:
        Features considered per split (default ``"sqrt"``).
    n_jobs:
        Accepted for compatibility; the ensemble always runs sequentially.
    seed:
        Master seed; each tree derives its own deterministic seed from it.
    max_bins:
        Histogram buckets used for split search.
    class_weight:
        Optional ``"balanced"`` class weighting.
    """

    name = "random_forest"
    _param_names = (
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "n_jobs",
        "seed",
        "max_bins",
        "class_weight",
    )

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[object] = "sqrt",
        n_jobs: int = 1,
        seed: int = DEFAULT_SEED,
        max_bins: int = 32,
        class_weight: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.n_jobs = int(n_jobs)
        self.seed = int(seed)
        self.max_bins = int(max_bins)
        self.class_weight = class_weight
        self._trees: List[DecisionTreeModel] = []
        self._oob: List[float] = []
        self._oob_mask: List[bool] = []
        self._train_y: List[int] = []

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "RandomForestModel":
        """Train the forest on bootstrap samples; returns ``self``."""
        Xi, yb, w = self._fit_prepare(
            X,
            y,
            feature_names=feature_names,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        n = len(Xi)
        self._train_y = yb
        self._trees = []
        if self._single_class:
            self._oob = [float(self._constant_prob or 0.0)] * n
            self._oob_mask = [True] * n
            self._fitted = True
            return self

        oob_sum = [0.0] * n
        oob_cnt = [0] * n
        for t in range(self.n_estimators):
            # Derive a distinct, reproducible seed per tree.
            tree_seed = self.seed + 1000 * t + 1
            rng = random.Random(tree_seed)
            boot = [rng.randrange(n) for _ in range(n)]
            in_bag = [False] * n
            for idx in boot:
                in_bag[idx] = True
            tree = DecisionTreeModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_bins=self.max_bins,
                seed=tree_seed,
                class_weight=self.class_weight,
            )
            tree.fit(
                [Xi[i] for i in boot],
                [yb[i] for i in boot],
                feature_names=self.feature_names,
                sample_weight=[w[i] for i in boot],
            )
            self._trees.append(tree)
            oob_idx = [i for i in range(n) if not in_bag[i]]
            if oob_idx:
                probs = tree.predict_proba([Xi[i] for i in oob_idx])
                for k, i in enumerate(oob_idx):
                    oob_sum[i] += probs[k]
                    oob_cnt[i] += 1

        self._oob = [
            (oob_sum[i] / oob_cnt[i]) if oob_cnt[i] > 0 else float("nan")
            for i in range(n)
        ]
        self._oob_mask = [oob_cnt[i] > 0 for i in range(n)]
        self._importance = self._average_importance()
        self._fitted = True
        return self

    def _average_importance(self) -> List[float]:
        """Average per-feature importance across trees, aligned to index."""
        if not self._trees:
            return [0.0] * self._n_features
        acc = [0.0] * self._n_features
        for tree in self._trees:
            imp = tree.feature_importance()
            for j, name in enumerate(self.feature_names):
                acc[j] += imp.get(name, 0.0)
        m = len(self._trees)
        return [v / m for v in acc]



    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Average the trees' class-1 probabilities (bagging)."""
        Xp = self._prepare_predict(X)
        if self._single_class and self._constant_prob is not None:
            return [float(self._constant_prob)] * len(Xp)
        if not self._trees:
            return [0.5] * len(Xp)
        sums = [0.0] * len(Xp)
        for tree in self._trees:
            probs = tree.predict_proba(Xp)
            for k in range(len(Xp)):
                sums[k] += probs[k]
        m = len(self._trees)
        return [s / m for s in sums]

    def oob_probabilities(self) -> List[float]:
        """Return out-of-bag ``P(class = 1)`` for each training row.

        Rows that were never out-of-bag for any tree yield ``nan``.
        """
        if not self._fitted:
            raise ValueError("random_forest: model must be fitted first")
        return list(self._oob)

    def oob_score(self) -> float:
        """Return out-of-bag accuracy over rows that were OOB at least once."""
        if not self._fitted:
            raise ValueError("random_forest: model must be fitted first")
        idx = [i for i in range(len(self._oob)) if self._oob_mask[i]]
        if not idx:
            return float("nan")
        correct = 0
        for i in idx:
            pred = 1 if self._oob[i] >= 0.5 else 0
            if pred == self._train_y[i]:
                correct += 1
        return correct / len(idx)
