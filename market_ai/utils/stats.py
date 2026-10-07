"""Dependency-free statistical helpers used across the system.

Every function here is deterministic and pure.  Nothing performs I/O and
nothing depends on the current time, which keeps backtests reproducible.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "mean", "median", "stdev", "variance", "percentile", "quantile",
    "zscore", "safe_div", "clamp", "correlation", "wilson_interval",
    "normal_cdf", "binomial_p_value", "expected_calibration_error",
    "brier_score", "log_loss", "profit_factor", "max_drawdown",
    "longest_losing_streak", "expectancy", "sharpe_like", "accuracy",
    "precision", "recall", "f1_score", "contingency",
]


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises and never returns inf/nan."""
    try:
        if denominator == 0 or not math.isfinite(denominator):
            return default
        result = numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default
    return result if math.isfinite(result) else default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``, mapping NaN to ``low``."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(fval):
        return low
    if fval < low:
        return low
    if fval > high:
        return high
    return fval


def _finite(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out.append(fv)
    return out


def mean(values: Sequence[float]) -> float:
    vals = _finite(values)
    return sum(vals) / len(vals) if vals else float("nan")


def median(values: Sequence[float]) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def variance(values: Sequence[float], ddof: int = 0) -> float:
    vals = _finite(values)
    n = len(vals)
    if n - ddof <= 0:
        return float("nan")
    mu = sum(vals) / n
    return sum((v - mu) ** 2 for v in vals) / (n - ddof)


def stdev(values: Sequence[float], ddof: int = 0) -> float:
    var = variance(values, ddof=ddof)
    return math.sqrt(var) if math.isfinite(var) and var >= 0 else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile; ``q`` in [0, 100]."""
    vals = sorted(_finite(values))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    q = clamp(q, 0.0, 100.0)
    pos = (len(vals) - 1) * (q / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return vals[int(pos)]
    frac = pos - lower
    return vals[lower] * (1.0 - frac) + vals[upper] * frac


def quantile(values: Sequence[float], q: float) -> float:
    """Quantile with ``q`` in [0, 1]."""
    return percentile(values, clamp(q, 0.0, 1.0) * 100.0)


def zscore(value: float, values: Sequence[float]) -> float:
    """Z-score of ``value`` against a reference sample."""
    mu = mean(values)
    sd = stdev(values, ddof=0)
    if not math.isfinite(mu) or not math.isfinite(sd) or sd <= 0:
        return 0.0
    return (value - mu) / sd


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation over pairwise-finite observations."""
    pairs = [
        (float(a), float(b))
        for a, b in zip(xs, ys)
        if _is_finite(a) and _is_finite(b)
    ]
    if len(pairs) < 2:
        return float("nan")
    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    mx, my = sum(xv) / len(xv), sum(yv) / len(yv)
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xv))
    sy = math.sqrt(sum((b - my) ** 2 for b in yv))
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def wilson_interval(successes: int, trials: int, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    Used everywhere a win rate is reported, so that small samples are
    visibly uncertain rather than presented as a point estimate.
    """
    if trials <= 0:
        return (0.0, 1.0)
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = phat + z * z / (2 * trials)
    margin = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
    return (clamp((centre - margin) / denom), clamp((centre + margin) / denom))


def normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def binomial_p_value(successes: int, trials: int, p0: float = 0.5, tail: str = "greater") -> float:
    """Exact one/two-sided binomial p-value.

    Implemented with a stable recurrence over log-probabilities so that
    reasonably large ``trials`` do not overflow.
    """
    if trials <= 0:
        return 1.0
    successes = int(clamp(successes, 0, trials))
    p0 = clamp(p0, 1e-9, 1 - 1e-9)

    def log_pmf(k: int) -> float:
        return (
            _log_comb(trials, k)
            + k * math.log(p0)
            + (trials - k) * math.log(1.0 - p0)
        )

    if tail == "greater":
        total = sum(math.exp(log_pmf(k)) for k in range(successes, trials + 1))
    elif tail == "less":
        total = sum(math.exp(log_pmf(k)) for k in range(0, successes + 1))
    else:
        observed = math.exp(log_pmf(successes))
        total = sum(
            math.exp(log_pmf(k))
            for k in range(0, trials + 1)
            if math.exp(log_pmf(k)) <= observed + 1e-12
        )
    return clamp(total)


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> float:
    """Expected Calibration Error over equal-width probability bins.

    A model that says 0.7 and is right 0.7 of the time has ECE ~ 0.
    """
    pairs = [
        (clamp(float(p)), 1 if float(o) >= 0.5 else 0)
        for p, o in zip(probabilities, outcomes)
        if _is_finite(p) and _is_finite(o)
    ]
    if not pairs:
        return float("nan")
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))
    total = len(pairs)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(p for p, _ in bucket) / len(bucket)
        avg_acc = sum(o for _, o in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_acc - avg_conf)
    return ece


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of probabilistic forecasts (lower is better)."""
    pairs = [
        (clamp(float(p)), 1.0 if float(o) >= 0.5 else 0.0)
        for p, o in zip(probabilities, outcomes)
        if _is_finite(p) and _is_finite(o)
    ]
    if not pairs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def log_loss(probabilities: Sequence[float], outcomes: Sequence[int], eps: float = 1e-12) -> float:
    """Binary cross-entropy (lower is better)."""
    pairs = [
        (clamp(float(p), eps, 1 - eps), 1.0 if float(o) >= 0.5 else 0.0)
        for p, o in zip(probabilities, outcomes)
        if _is_finite(p) and _is_finite(o)
    ]
    if not pairs:
        return float("nan")
    return -sum(o * math.log(p) + (1 - o) * math.log(1 - p) for p, o in pairs) / len(pairs)


def profit_factor(pnls: Sequence[float]) -> float:
    """Gross profit / gross loss.  ``inf`` when there are no losses."""
    gains = sum(p for p in pnls if _is_finite(p) and p > 0)
    losses = -sum(p for p in pnls if _is_finite(p) and p < 0)
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def expectancy(pnls: Sequence[float]) -> float:
    """Average profit/loss per trade.  The headline profitability metric."""
    vals = _finite(pnls)
    return sum(vals) / len(vals) if vals else float("nan")


def max_drawdown(equity: Sequence[float]) -> Tuple[float, float]:
    """Return ``(absolute_drawdown, relative_drawdown)`` for an equity curve."""
    vals = _finite(equity)
    if not vals:
        return (0.0, 0.0)
    peak = vals[0]
    worst_abs = 0.0
    worst_rel = 0.0
    for value in vals:
        if value > peak:
            peak = value
        drop = peak - value
        if drop > worst_abs:
            worst_abs = drop
        if peak > 0:
            rel = drop / peak
            if rel > worst_rel:
                worst_rel = rel
    return (worst_abs, worst_rel)


def longest_losing_streak(pnls: Sequence[float]) -> int:
    """Longest run of strictly negative outcomes."""
    best = current = 0
    for p in pnls:
        if _is_finite(p) and p < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def sharpe_like(pnls: Sequence[float]) -> float:
    """Per-trade Sharpe-style ratio (mean / stdev).  Not annualised."""
    vals = _finite(pnls)
    if len(vals) < 2:
        return float("nan")
    sd = stdev(vals, ddof=1)
    if not math.isfinite(sd) or sd <= 0:
        return float("nan")
    return (sum(vals) / len(vals)) / sd


def accuracy(predicted: Sequence[int], actual: Sequence[int]) -> float:
    pairs = [(p, a) for p, a in zip(predicted, actual) if _is_finite(p) and _is_finite(a)]
    if not pairs:
        return float("nan")
    return sum(1 for p, a in pairs if int(p) == int(a)) / len(pairs)


def precision(predicted_positive: Sequence[int], actual_positive: Sequence[int]) -> float:
    """TP / (TP + FP); NaN when the model made no positive calls."""
    tp = fp = 0
    for p, a in zip(predicted_positive, actual_positive):
        if int(p) == 1:
            if int(a) == 1:
                tp += 1
            else:
                fp += 1
    return tp / (tp + fp) if (tp + fp) else float("nan")


def recall(predicted_positive: Sequence[int], actual_positive: Sequence[int]) -> float:
    """TP / (TP + FN); NaN when there are no actual positives."""
    tp = fn = 0
    for p, a in zip(predicted_positive, actual_positive):
        if int(a) == 1:
            if int(p) == 1:
                tp += 1
            else:
                fn += 1
    return tp / (tp + fn) if (tp + fn) else float("nan")


def f1_score(predicted_positive: Sequence[int], actual_positive: Sequence[int]) -> float:
    prec = precision(predicted_positive, actual_positive)
    rec = recall(predicted_positive, actual_positive)
    if not (math.isfinite(prec) and math.isfinite(rec)) or (prec + rec) == 0:
        return float("nan")
    return 2 * prec * rec / (prec + rec)


def contingency(predicted: Sequence[int], actual: Sequence[int]) -> Dict[str, int]:
    """Confusion-matrix counts for binary labels."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for p, a in zip(predicted, actual):
        pv, av = int(p), int(a)
        if pv == 1 and av == 1:
            counts["tp"] += 1
        elif pv == 1 and av == 0:
            counts["fp"] += 1
        elif pv == 0 and av == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts
