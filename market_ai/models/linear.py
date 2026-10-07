"""L2-regularised logistic regression in pure Python.

Trained by batch gradient descent on the (optionally class-balanced) logistic
loss.  Features are z-score standardised internally so a single global learning
rate converges quickly and the L2 penalty is scale-invariant.  Determinism is
guaranteed: the optimiser is fully deterministic, so the seed only exists for
API symmetry with the tree-based models.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from ..config import DEFAULT_SEED
from .base import BaseModel, sigmoid, standardize_apply, standardize_fit

__all__ = ["LogisticRegressionModel"]


class LogisticRegressionModel(BaseModel):
    """Binary logistic regression with L2 regularisation.

    Parameters
    ----------
    learning_rate:
        Step size for batch gradient descent.
    epochs:
        Number of full-batch passes over the training data.
    l2:
        L2 penalty strength applied to the weights (never to the bias).
    class_weight:
        ``"balanced"`` re-weights classes inversely to their frequency.
    seed:
        Retained for API symmetry; the fit itself is deterministic.
    standardize:
        Z-score features internally (strongly recommended for gradient descent).
    """

    name = "logistic"
    _param_names = (
        "learning_rate",
        "epochs",
        "l2",
        "class_weight",
        "seed",
        "standardize",
    )

    def __init__(
        self,
        learning_rate: float = 0.5,
        epochs: int = 400,
        l2: float = 0.01,
        class_weight: Optional[str] = "balanced",
        seed: int = DEFAULT_SEED,
        standardize: bool = True,
    ) -> None:
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.seed = int(seed)
        self.standardize = bool(standardize)
        self._weights: List[float] = []
        self._bias: float = 0.0
        self._means: List[float] = []
        self._stds: List[float] = []

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "LogisticRegressionModel":
        """Fit the model by batch gradient descent; returns ``self``."""
        Xi, yb, weights = self._fit_prepare(
            X,
            y,
            feature_names=feature_names,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        d = self._n_features
        if self._single_class:
            # Degenerate target: a constant equal to the class prior is both
            # the maximum-likelihood and the best-calibrated prediction.
            self._weights = [0.0] * d
            self._bias = 0.0
            self._means = [0.0] * d
            self._stds = [1.0] * d
            self._fitted = True
            return self

        if self.standardize:
            self._means, self._stds = standardize_fit(Xi)
            Xs = standardize_apply(Xi, self._means, self._stds)
        else:
            self._means = [0.0] * d
            self._stds = [1.0] * d
            Xs = Xi

        self._weights = [0.0] * d
        self._bias = 0.0
        self._gradient_descent(Xs, yb, weights)
        total = sum(abs(w) for w in self._weights)
        # Normalise to a unit sum so importance is comparable across models.
        self._importance = (
            [abs(w) / total for w in self._weights] if total > 0 else [0.0] * d
        )
        self._fitted = True
        return self

    def _gradient_descent(
        self, Xs: List[List[float]], yb: List[int], weights: List[float]
    ) -> None:
        """Run full-batch gradient descent on the weighted logistic loss.

        Weights are normalised to sum to one so the effective learning rate is
        independent of the training-set size.
        """
        n = len(Xs)
        total = sum(weights) or 1.0
        wn = [w / total for w in weights]
        d = self._n_features
        lr = self.learning_rate
        l2 = self.l2
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                xi = Xs[i]
                z = self._bias
                for j in range(d):
                    z += self._weights[j] * xi[j]
                err = (sigmoid(z) - yb[i]) * wn[i]
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            max_grad = abs(gb)
            for j in range(d):
                gw[j] += l2 * self._weights[j]  # L2 applies to weights only
                self._weights[j] -= lr * gw[j]
                if abs(gw[j]) > max_grad:
                    max_grad = abs(gw[j])
            self._bias -= lr * gb
            # Early stop once the gradient is negligible; saves time without
            # changing the converged solution.
            if max_grad < 1e-9:
                break

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Return ``P(class = 1)`` for each row, guaranteed in ``[0, 1]``."""
        Xp = self._prepare_predict(X)
        if self._single_class and self._constant_prob is not None:
            return [float(self._constant_prob)] * len(Xp)
        if self.standardize:
            Xp = standardize_apply(Xp, self._means, self._stds)
        out: List[float] = []
        for xi in Xp:
            z = self._bias
            for j in range(self._n_features):
                z += self._weights[j] * xi[j]
            out.append(sigmoid(z))
        return out

