"""Probability calibration and reliability reporting.

Two classical calibrators are provided:

* ``method="platt"`` - a 1-D logistic regression ``sigmoid(a*p + b)`` fitted by
  gradient descent on the raw scores.
* ``method="isotonic"`` - a non-decreasing step function fitted with the
  Pool-Adjacent-Violators Algorithm (PAVA) and evaluated by interpolation.

Both are pure Python and fully deterministic.  ``calibration_report`` summarises
a set of probabilities with a reliability curve, ECE, Brier and log loss,
reusing the shared statistics helpers so the numbers match the rest of the
system.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..utils.stats import brier_score as _brier
from ..utils.stats import clamp as _clamp
from ..utils.stats import expected_calibration_error as _ece
from ..utils.stats import log_loss as _log_loss
from .base import sigmoid, to_float

__all__ = ["Calibrator", "calibration_report"]


class Calibrator:
    """Map raw scores to calibrated probabilities.

    Parameters
    ----------
    method:
        ``"platt"`` (logistic) or ``"isotonic"`` (PAVA step function).
    epochs / learning_rate / l2:
        Optimiser settings for the Platt fit (ignored by isotonic).
    """

    def __init__(
        self,
        method: str = "platt",
        *,
        epochs: int = 500,
        learning_rate: float = 0.5,
        l2: float = 1e-6,
    ) -> None:
        if method not in ("platt", "isotonic"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self._fitted = False
        self._a = 1.0
        self._b = 0.0
        self._x: List[float] = []
        self._y: List[float] = []

    def fit(self, probs: Sequence[float], y: Sequence[int]) -> "Calibrator":
        """Fit the calibrator on raw ``probs`` and binary ``y``; returns self."""
        p = [_clamp(to_float(v)) for v in probs]
        yy = [1 if to_float(v) >= 0.5 else 0 for v in y]
        if len(p) != len(yy):
            raise ValueError("probs and y must have the same length")
        if not p:
            raise ValueError("probs must be non-empty")
        if self.method == "platt":
            self._fit_platt(p, yy)
        else:
            self._fit_isotonic(p, yy)
        self._fitted = True
        return self

    def _fit_platt(self, p: List[float], y: List[int]) -> None:
        """Fit ``a`` and ``b`` by batch gradient descent on the BCE loss."""
        a, b = 1.0, 0.0
        n = len(p)
        for _ in range(self.epochs):
            grad_a = 0.0
            grad_b = 0.0
            for i in range(n):
                err = sigmoid(a * p[i] + b) - y[i]
                grad_a += err * p[i]
                grad_b += err
            grad_a = grad_a / n + self.l2 * a
            grad_b = grad_b / n
            a -= self.learning_rate * grad_a
            b -= self.learning_rate * grad_b
        self._a = a
        self._b = b

    def _fit_isotonic(self, p: List[float], y: List[int]) -> None:
        """Fit a non-decreasing step function with PAVA."""
        order = sorted(range(len(p)), key=lambda i: p[i])
        blocks: List[List[float]] = []  # each block: [sum_y, count, max_p]
        for i in order:
            blocks.append([float(y[i]), 1.0, p[i]])
            # Merge adjacent blocks that violate monotonicity.
            while len(blocks) >= 2:
                prev, cur = blocks[-2], blocks[-1]
                if prev[0] / prev[1] > cur[0] / cur[1]:
                    blocks.pop()
                    blocks.pop()
                    blocks.append(
                        [prev[0] + cur[0], prev[1] + cur[1], max(prev[2], cur[2])]
                    )
                else:
                    break
        xs: List[float] = []
        ys: List[float] = []
        for block in blocks:
            xv = block[2]
            yv = block[0] / block[1]
            if xs and xv <= xs[-1]:
                # Ties at a boundary: keep x strictly increasing, take the
                # later (>=) value to preserve monotonicity.
                ys[-1] = max(ys[-1], yv)
                continue
            xs.append(xv)
            ys.append(yv)
        if not xs:  # defensive: degenerate input
            xs, ys = [0.0, 1.0], [0.0, 1.0]
        self._x = xs
        self._y = ys
