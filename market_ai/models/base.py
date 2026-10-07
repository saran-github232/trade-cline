"""Pure-Python supervised-learning primitives and shared plumbing.

This module defines the frozen :class:`Model` contract from ``INTERFACES.md``
section 6 together with the small numeric helpers every model reuses.  It is
deliberately dependency-free (standard library only) so the whole modelling
stack is reproducible and importable in a bare Python 3.11 environment.

Design notes
------------
* Missing values are **data**, never a programmer error.  They are turned into
  ``float('nan')`` and imputed, while shape mismatches *are* programmer errors
  and raise :class:`ValueError`.
* Every model is deterministic given a ``seed``; no module-level ``random``
  state is ever touched (we always build a local ``random.Random``).
* Column statistics ignore non-finite entries, so a column made entirely of
  ``nan`` degrades to a neutral filler instead of poisoning the whole fit.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..config import DEFAULT_SEED
from ..utils.stats import mean as _mean
from ..utils.stats import median as _median
from ..utils.stats import stdev as _stdev

__all__ = [
    "Model",
    "BaseModel",
    "impute",
    "standardize_fit",
    "standardize_apply",
    "train_test_time_split",
    "predict_labels",
    "to_float",
]

_NAN = float("nan")


# --------------------------------------------------------------------------
# Low level numeric helpers
# --------------------------------------------------------------------------
def to_float(value: Any) -> float:
    """Best-effort conversion to ``float``.

    ``None`` and anything non-numeric become ``float('nan')`` rather than
    raising, because a missing feature value is a data-quality issue that the
    imputation step is responsible for, not a programming mistake.
    """
    if value is None:
        return _NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


def _rectangular(X: Sequence[Sequence[float]]) -> List[List[float]]:
    """Convert ``X`` to a rectangular list-of-lists of floats.

    Raises :class:`ValueError` on ragged input because a ragged matrix is a
    programmer error (bad shape), not bad data.
    """
    rows: List[List[float]] = []
    width: Optional[int] = None
    for i, row in enumerate(X):
        try:
            values = list(row)
        except TypeError as exc:  # not a sequence at all
            raise ValueError(f"row {i} is not a sequence") from exc
        if width is None:
            width = len(values)
            if width == 0:
                raise ValueError("rows must contain at least one feature")
        elif len(values) != width:
            raise ValueError(
                f"row {i} has {len(values)} features, expected {width}"
            )
        rows.append([to_float(v) for v in values])
    return rows


def _column_filler(column: Sequence[float], strategy: str) -> float:
    """Return the imputation value for one column under ``strategy``."""
    finite = [v for v in column if math.isfinite(v)]
    if strategy == "median":
        filler = _median(finite)
    elif strategy == "mean":
        filler = _mean(finite)
    elif strategy == "zero":
        filler = 0.0
    else:  # pragma: no cover - guarded by callers, kept for safety
        raise ValueError(f"unknown imputation strategy {strategy!r}")
    return filler if math.isfinite(filler) else 0.0


def sigmoid(z: float) -> float:
    """Numerically stable logistic function.

    The two-branch form avoids ``exp`` overflow for large ``|z|`` and always
    returns a value in the open interval ``(0, 1)``.
    """
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)




# --------------------------------------------------------------------------
# Frozen helper API (INTERFACES.md section 6)
# --------------------------------------------------------------------------
def impute(X: Sequence[Sequence[float]], strategy: str = "median") -> List[List[float]]:
    """Replace non-finite feature values with a per-column statistic.

    ``strategy`` is one of ``"median"`` (default), ``"mean"`` or ``"zero"``.
    The filler is computed from the finite values of each column; a column with
    no finite value falls back to ``0.0`` so downstream maths stays defined.
    """
    rows = _rectangular(X)
    if not rows:
        return rows
    n_cols = len(rows[0])
    fillers = [
        _column_filler([r[j] for r in rows], strategy) for j in range(n_cols)
    ]
    return [
        [r[j] if math.isfinite(r[j]) else fillers[j] for j in range(n_cols)]
        for r in rows
    ]


def standardize_fit(
    X: Sequence[Sequence[float]],
) -> Tuple[List[float], List[float]]:
    """Return per-column ``(means, stds)`` for z-score standardisation.

    Non-finite values are ignored when estimating the statistics.  A column
    with zero (or non-finite) spread gets ``std = 1.0`` so that
    :func:`standardize_apply` never divides by zero.
    """
    rows = _rectangular(X)
    if not rows:
        raise ValueError("X must contain at least one row")
    n_cols = len(rows[0])
    means: List[float] = []
    stds: List[float] = []
    for j in range(n_cols):
        finite = [r[j] for r in rows if math.isfinite(r[j])]
        mu = _mean(finite) if finite else 0.0
        sd = _stdev(finite, ddof=0) if finite else _NAN
        if not math.isfinite(mu):
            mu = 0.0
        if not math.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        means.append(mu)
        stds.append(sd)
    return means, stds


def standardize_apply(
    X: Sequence[Sequence[float]], means: Sequence[float], stds: Sequence[float]
) -> List[List[float]]:
    """Apply a previously fitted z-score transform; non-finite maps to ``0.0``.

    Mapping a missing value to the column mean (``0.0`` after standardisation)
    is the neutral choice for a linear model and keeps the output finite.
    """
    out: List[List[float]] = []
    for row in X:
        transformed: List[float] = []
        for j, value in enumerate(row):
            fv = to_float(value)
            if not math.isfinite(fv):
                transformed.append(0.0)
            else:
                transformed.append((fv - means[j]) / stds[j])
        out.append(transformed)
    return out


def train_test_time_split(
    X: Sequence[Sequence[float]], y: Sequence[int], frac: float
) -> Tuple[List, List, List, List]:
    """Split ``(X, y)`` on a **time-ordered** boundary, never shuffling.

    ``frac`` is the fraction of the *earliest* rows assigned to the training
    set.  Shuffling would leak future information into training, so the split
    is strictly positional.  When ``2 <= n`` the boundary is clamped so both
    sides are non-empty.  Returns ``(X_train, y_train, X_test, y_test)``.
    """
    Xl = list(X)
    yl = list(y)
    if len(Xl) != len(yl):
        raise ValueError("X and y must have the same number of rows")
    n = len(Xl)
    if not (0.0 < frac < 1.0):
        raise ValueError("frac must be in the open interval (0, 1)")
    if n == 0:
        return [], [], [], []
    if n == 1:
        return Xl, yl, [], []
    split = int(n * frac)
    split = max(1, min(n - 1, split))
    return Xl[:split], yl[:split], Xl[split:], yl[split:]


def predict_labels(probs: Sequence[float], threshold: float = 0.5) -> List[int]:
    """Turn probabilities into 0/1 labels using ``>= threshold``.

    ``nan`` probabilities cannot exceed the threshold and therefore fall to
    class ``0`` - a deliberately conservative default.
    """
    return [1 if to_float(p) >= threshold else 0 for p in probs]



# --------------------------------------------------------------------------
# The frozen Model contract
# --------------------------------------------------------------------------
@runtime_checkable
class Model(Protocol):
    """Structural contract every model in this package satisfies."""

    name: str

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "Model":
        """Train on ``X``/``y`` and return ``self``."""

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Return ``P(class = 1)`` for each row of ``X``."""

    def feature_importance(self) -> Dict[str, float]:
        """Return a mapping of feature name to non-negative importance."""

    def get_params(self) -> Dict[str, Any]:
        """Return the constructor hyper-parameters as a plain dict."""

    def set_params(self, **params: Any) -> "Model":
        """Update hyper-parameters in place and return ``self``."""


# --------------------------------------------------------------------------
# Shared plumbing for concrete models
# --------------------------------------------------------------------------
class BaseModel:
    """Shared plumbing: feature names, imputation, weights, shape checks.

    Concrete models only need to implement :meth:`fit` and
    :meth:`predict_proba`; everything else (feature-name bookkeeping, NaN
    imputation, class-balance weights, importance formatting, parameter
    introspection) lives here.
    """

    name: str = "base"
    #: Hyper-parameters surfaced by :meth:`get_params`; subclasses override.
    _param_names: Tuple[str, ...] = ()

    def __init__(self) -> None:
        self.feature_names: List[str] = []
        self.impute_strategy: str = "median"
        self.class_weight: Optional[str] = None
        self._fitted: bool = False
        self._n_features: int = 0
        self._impute_values: List[float] = []
        self._importance: List[float] = []
        self._single_class: bool = False
        self._constant_prob: Optional[float] = None

    # -- parameter introspection ------------------------------------------
    def get_params(self) -> Dict[str, Any]:
        """Return the declared hyper-parameters as a plain dict."""
        return {name: getattr(self, name) for name in self._param_names}

    def set_params(self, **params: Any) -> "BaseModel":
        """Set hyper-parameters by name; unknown names raise ``ValueError``."""
        for key, value in params.items():
            if key not in self._param_names:
                raise ValueError(f"{self.name}: unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    # -- validation / preprocessing ---------------------------------------
    def _check_X(
        self, X: Sequence[Sequence[float]], n_features: Optional[int] = None
    ) -> List[List[float]]:
        """Validate ``X`` shape and coerce to a rectangular list-of-lists.

        Empty input and ragged rows are programmer errors and raise
        ``ValueError``; individual non-numeric cells become ``nan``.
        """
        rows = _rectangular(X)
        if not rows:
            raise ValueError("X must contain at least one row")
        width = len(rows[0])
        if n_features is not None and width != n_features:
            raise ValueError(f"X has {width} features, expected {n_features}")
        return rows

    def _resolve_feature_names(
        self, feature_names: Optional[Sequence[str]]
    ) -> List[str]:
        """Return the feature names, validating the count when supplied."""
        if feature_names is None:
            return [f"f{i}" for i in range(self._n_features)]
        names = [str(f) for f in feature_names]
        if len(names) != self._n_features:
            raise ValueError(
                f"feature_names has {len(names)} entries, "
                f"expected {self._n_features}"
            )
        return names

    def _compute_impute_values(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Compute per-column fillers from the (already validated) ``X``."""
        if not X:
            return []
        n_cols = len(X[0])
        return [
            _column_filler([r[j] for r in X], self.impute_strategy)
            for j in range(n_cols)
        ]

    def _apply_impute(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Replace non-finite cells using the fillers learned during ``fit``."""
        out: List[List[float]] = []
        for row in X:
            filled: List[float] = []
            for j, value in enumerate(row):
                if math.isfinite(value):
                    filled.append(value)
                elif j < len(self._impute_values):
                    filled.append(self._impute_values[j])
                else:
                    filled.append(0.0)
            out.append(filled)
        return out

    def _resolve_weights(
        self,
        y: Sequence[int],
        sample_weight: Optional[Sequence[float]],
        class_weight: Optional[str],
    ) -> List[float]:
        """Return per-sample weights honouring explicit or balanced weighting.

        ``sample_weight`` wins when provided.  Otherwise ``"balanced"`` scales
        each class so both contribute equally to the loss - important when the
        positive class is rare.
        """
        n = len(y)
        if sample_weight is not None:
            weights = [max(0.0, to_float(v)) for v in sample_weight]
            if len(weights) != n:
                raise ValueError("sample_weight length must match y")
            if sum(weights) <= 0.0:
                return [1.0] * n
            return weights
        if class_weight == "balanced" and n > 0:
            pos = sum(1 for v in y if v == 1)
            neg = n - pos
            if pos > 0 and neg > 0:
                w_pos = n / (2.0 * pos)
                w_neg = n / (2.0 * neg)
                return [w_pos if v == 1 else w_neg for v in y]
        return [1.0] * n

    def _fit_prepare(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_names: Optional[Sequence[str]],
        sample_weight: Optional[Sequence[float]],
        class_weight: Optional[str],
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        """Common ``fit`` prologue shared by every concrete model.

        Validates shapes, stores feature names and imputation fillers, detects
        the single-class edge case, and produces class-balanced weights.
        Returns the imputed ``X``, binarised ``y`` and the weights.
        """
        Xv = self._check_X(X)
        if len(Xv) != len(y):
            raise ValueError("X and y must have the same number of rows")
        self._n_features = len(Xv[0])
        self.feature_names = self._resolve_feature_names(feature_names)
        self._impute_values = self._compute_impute_values(Xv)
        Xi = self._apply_impute(Xv)

        yb = [1 if to_float(v) >= 0.5 else 0 for v in y]
        self._single_class = len(set(yb)) < 2
        # A single-class target is degenerate: predict the class prior exactly
        # instead of trying to fit a decision boundary that cannot exist.
        self._constant_prob = (
            (sum(yb) / len(yb)) if (self._single_class and yb) else None
        )
        weights = self._resolve_weights(yb, sample_weight, class_weight)
        self._importance = [0.0] * self._n_features
        return Xi, yb, weights

    def _prepare_predict(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        """Validate ``X`` for prediction and apply learned imputation."""
        if not self._fitted:
            raise ValueError(f"{self.name}: model must be fitted before predicting")
        Xv = self._check_X(X, n_features=self._n_features)
        return self._apply_impute(Xv)

    # -- shared outputs ----------------------------------------------------
    def feature_importance(self) -> Dict[str, float]:
        """Return per-feature importance keyed by feature name.

        Concrete models fill ``self._importance`` (aligned to feature index);
        this method maps it to names.  An untrained model returns zeros.
        """
        names = self.feature_names or [
            f"f{i}" for i in range(len(self._importance))
        ]
        return {
            names[j]: float(self._importance[j])
            for j in range(len(self._importance))
        }

    def predict(
        self, X: Sequence[Sequence[float]], threshold: float = 0.5
    ) -> List[int]:
        """Convenience wrapper returning 0/1 labels from :meth:`predict_proba`."""
        return predict_labels(self.predict_proba(X), threshold)


