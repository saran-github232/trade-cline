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

    def transform(self, probs: Sequence[float]) -> List[float]:
        """Return calibrated probabilities for raw ``probs``."""
        if not self._fitted:
            raise ValueError("Calibrator must be fitted before transform")
        out: List[float] = []
        for v in probs:
            p = _clamp(to_float(v))
            if self.method == "platt":
                out.append(sigmoid(self._a * p + self._b))
            else:
                out.append(self._interp(p))
        return out

    def _interp(self, p: float) -> float:
        """Piecewise-linear interpolation of the isotonic fit, clamped at ends."""
        xs, ys = self._x, self._y
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        lo = bisect_right(xs, p) - 1
        if lo < 0:
            return ys[0]
        if lo >= len(xs) - 1:
            return ys[-1]
        x0, x1 = xs[lo], xs[lo + 1]
        y0, y1 = ys[lo], ys[lo + 1]
        if x1 <= x0:
            return y1
        t = (p - x0) / (x1 - x0)
        return _clamp(y0 + (y1 - y0) * t)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the calibrator to a plain, pickle-free dict."""
        payload: Dict[str, Any] = {"method": self.method, "fitted": self._fitted}
        if self.method == "platt":
            payload["a"] = self._a
            payload["b"] = self._b
        else:
            payload["x"] = list(self._x)
            payload["y"] = list(self._y)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Calibrator":
        """Rebuild a calibrator from :meth:`to_dict` output."""
        calibrator = cls(method=str(payload.get("method", "platt")))
        if calibrator.method == "platt":
            calibrator._a = float(payload.get("a", 1.0))
            calibrator._b = float(payload.get("b", 0.0))
        else:
            calibrator._x = [float(v) for v in payload.get("x", [])]
            calibrator._y = [float(v) for v in payload.get("y", [])]
        calibrator._fitted = bool(payload.get("fitted", True))
        return calibrator


def calibration_report(
    probs: Sequence[float], y: Sequence[int], n_bins: int = 10
) -> Dict[str, Any]:
    """Summarise calibration quality with a reliability curve and scores.

    Returns a dict with ``bins`` (one entry per equal-width probability bucket,
    each holding its count, average confidence, average outcome and gap), plus
    ``ece``, ``brier``, ``log_loss`` and the sample count ``n``.  The three
    scalar metrics are delegated to :mod:`market_ai.utils.stats` so they are
    consistent with every other report in the system.
    """
    ps = [_clamp(to_float(v)) for v in probs]
    ys = [1 if to_float(v) >= 0.5 else 0 for v in y]
    if len(ps) != len(ys):
        raise ValueError("probs and y must have the same length")
    n_bins = max(1, int(n_bins))
    pairs = list(zip(ps, ys))
    buckets: List[List] = [[] for _ in range(n_bins)]
    for p, o in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        buckets[idx].append((p, o))

    bins: List[Dict[str, Any]] = []
    for b, bucket in enumerate(buckets):
        if bucket:
            avg_prob = sum(p for p, _ in bucket) / len(bucket)
            avg_out = sum(o for _, o in bucket) / len(bucket)
            gap = abs(avg_out - avg_prob)
        else:
            avg_prob = avg_out = gap = float("nan")
        bins.append(
            {
                "index": b,
                "lower": b / n_bins,
                "upper": (b + 1) / n_bins,
                "count": len(bucket),
                "avg_prob": avg_prob,
                "avg_outcome": avg_out,
                "gap": gap,
            }
        )

    return {
        "bins": bins,
        "ece": _ece(ps, ys, n_bins),
        "brier": _brier(ps, ys),
        "log_loss": _log_loss(ps, ys),
        "n": len(pairs),
    }

