"""Analytics over accumulated experience - the error-learning layer.

Everything here answers a question a human would ask about the system's own
mistakes: *where* is it wrong, *when* is it overconfident, and *which*
recurring pattern keeps costing money.  Nothing here mutates rules; results
feed the training dataset and the retraining policy, and any rule change must
still pass Phase 7 backtesting before it becomes production behaviour.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..types import Decision, ErrorCategory, Experience, Regime
from ..utils.stats import (
    clamp,
    expected_calibration_error,
    mean,
    safe_div,
    wilson_interval,
)

__all__ = [
    "error_breakdown",
    "regime_failures",
    "confidence_reliability",
    "recurring_patterns",
    "timeframe_failures",
    "accuracy_trend",
    "summarise",
]

_DIRECTIONAL_ERRORS = (ErrorCategory.FALSE_UP, ErrorCategory.FALSE_DOWN)
_CORRECT_DIRECTIONAL = (ErrorCategory.CORRECT_UP, ErrorCategory.CORRECT_DOWN)


def _directional(items: Sequence[Experience]) -> List[Experience]:
    """Experiences where real money was at risk (UP/DOWN, not NO-TRADE)."""
    return [e for e in items if e.decision in (Decision.UP, Decision.DOWN)]


def error_breakdown(experiences: Sequence[Experience]) -> Dict[str, Any]:
    """Counts and rates for every :class:`ErrorCategory`."""
    total = len(experiences)
    counts: Dict[str, int] = {c.value: 0 for c in ErrorCategory}
    for experience in experiences:
        counts[experience.error_category.value] = counts.get(experience.error_category.value, 0) + 1
    directional = _directional(experiences)
    wins = sum(1 for e in directional if e.error_category in _CORRECT_DIRECTIONAL)
    losses = sum(1 for e in directional if e.error_category in _DIRECTIONAL_ERRORS)
    low, high = wilson_interval(wins, len(directional)) if directional else (0.0, 1.0)
    return {
        "total": total,
        "counts": counts,
        "rates": {k: (safe_div(v, total) if total else 0.0) for k, v in counts.items()},
        "directional_total": len(directional),
        "wins": wins,
        "losses": losses,
        "win_rate": safe_div(wins, len(directional), float("nan")),
        "win_rate_ci": [low, high],
        "no_trade_rate": safe_div(counts[ErrorCategory.NO_TRADE_CORRECT.value]
                                  + counts[ErrorCategory.NO_TRADE_MISSED.value], total),
        "missed_rate": safe_div(counts[ErrorCategory.NO_TRADE_MISSED.value], total),
    }


def regime_failures(experiences: Sequence[Experience]) -> Dict[str, Any]:
    """Per-regime precision and expectancy, with small-sample flags.

    This is the primary tool for spotting regime-specific failures - e.g. the
    model being fine in trends and consistently wrong in RANGE.
    """
    buckets: Dict[str, List[Experience]] = {}
    for experience in experiences:
        buckets.setdefault(experience.regime.value, []).append(experience)

    out: Dict[str, Any] = {}
    for regime, items in sorted(buckets.items()):
        directional = _directional(items)
        wins = sum(1 for e in directional if e.error_category in _CORRECT_DIRECTIONAL)
        pnls = [e.profit_or_loss for e in directional]
        low, high = wilson_interval(wins, len(directional)) if directional else (0.0, 1.0)
        out[regime] = {
            "n": len(items),
            "directional_n": len(directional),
            "precision": safe_div(wins, len(directional), float("nan")),
            "precision_ci": [low, high],
            "expectancy": mean(pnls) if pnls else float("nan"),
            "no_trade_rate": safe_div(
                sum(1 for e in items if e.decision is Decision.NO_TRADE), len(items)
            ),
            # Fewer than 30 directional samples cannot support a claim either way.
            "reliable": len(directional) >= 30,
        }
    return out


def confidence_reliability(experiences: Sequence[Experience], n_bins: int = 5) -> Dict[str, Any]:
    """Is stated confidence actually earned?

    Compares realised precision against mean stated confidence per bucket and
    reports Expected Calibration Error.  Systematic gaps here are exactly what
    OVERCONFIDENCE / UNDERCONFIDENCE tags are meant to surface.
    """
    directional = _directional(experiences)
    if not directional:
        return {"bins": [], "ece": float("nan"), "n": 0, "bias": "unknown"}

    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: List[Dict[str, Any]] = []
    probs: List[float] = []
    outcomes: List[int] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        members = [
            e for e in directional
            if lo <= e.confidence < hi or (i == n_bins - 1 and e.confidence == hi)
        ]
        if not members:
            continue
        wins = sum(1 for e in members if e.error_category in _CORRECT_DIRECTIONAL)
        avg_conf = mean([e.confidence for e in members])
        realised = safe_div(wins, len(members))
        bins.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "n": len(members),
            "mean_confidence": avg_conf,
            "realised_precision": realised,
            "gap": realised - avg_conf,
        })
        probs.extend(e.confidence for e in members)
        outcomes.extend(1 if e.error_category in _CORRECT_DIRECTIONAL else 0 for e in members)

    ece = expected_calibration_error(probs, outcomes, n_bins=max(n_bins, 5))
    overall_conf = mean(probs)
    overall_prec = safe_div(sum(outcomes), len(outcomes))
    bias = "unknown"
    if math.isfinite(overall_conf) and math.isfinite(overall_prec):
        if overall_conf - overall_prec > 0.08:
            bias = "overconfident"
        elif overall_prec - overall_conf > 0.08:
            bias = "underconfident"
        else:
            bias = "calibrated"
    return {
        "bins": bins,
        "ece": ece,
        "n": len(directional),
        "mean_confidence": overall_conf,
        "mean_precision": overall_prec,
        "bias": bias,
    }


def timeframe_failures(experiences: Sequence[Experience]) -> Dict[str, Any]:
    """Per-timeframe precision, so a bad timeframe can be switched off."""
    buckets: Dict[str, List[Experience]] = {}
    for experience in experiences:
        buckets.setdefault(experience.timeframe.value, []).append(experience)
    out: Dict[str, Any] = {}
    for tf, items in sorted(buckets.items()):
        directional = _directional(items)
        wins = sum(1 for e in directional if e.error_category in _CORRECT_DIRECTIONAL)
        out[tf] = {
            "n": len(items),
            "directional_n": len(directional),
            "precision": safe_div(wins, len(directional), float("nan")),
            "expectancy": mean([e.profit_or_loss for e in directional]) if directional else float("nan"),
            "reliable": len(directional) >= 30,
        }
    return out


def recurring_patterns(
    experiences: Sequence[Experience], *, min_support: int = 5
) -> List[Dict[str, Any]]:
    """Rank recurring failure patterns by how much they actually cost.

    Patterns are keyed by (error tag, regime) because the same tag can be
    benign in a trend and expensive in a range.  ``cost`` is the summed loss
    attributed to the pattern, which is what should drive prioritisation.
    """
    buckets: Dict[Tuple[str, str], List[Experience]] = {}
    for experience in experiences:
        if experience.error_category not in _DIRECTIONAL_ERRORS:
            continue
        for tag in experience.tags:
            if tag in ("GOOD_ABSTENTION",):
                continue
            buckets.setdefault((tag, experience.regime.value), []).append(experience)

    patterns: List[Dict[str, Any]] = []
    for (tag, regime), items in buckets.items():
        if len(items) < min_support:
            continue
        losses = [e.profit_or_loss for e in items if e.profit_or_loss < 0]
        patterns.append({
            "pattern": tag,
            "regime": regime,
            "support": len(items),
            "cost": abs(sum(losses)),
            "avg_loss": mean(losses) if losses else 0.0,
            "mean_confidence": mean([e.confidence for e in items]),
            "rate": safe_div(len(items), len(experiences)),
        })
    patterns.sort(key=lambda p: (-p["cost"], -p["support"]))
    return patterns


def accuracy_trend(
    experiences: Sequence[Experience], *, windows: int = 5
) -> Dict[str, Any]:
    """Precision over successive equal-sized windows (degradation detection).

    A downward drift here is the signal the retraining policy listens for.
    """
    directional = sorted(_directional(experiences), key=lambda e: e.timestamp)
    if len(directional) < windows * 2:
        return {"windows": [], "slope": 0.0, "degrading": False, "n": len(directional)}
    size = len(directional) // windows
    out: List[Dict[str, Any]] = []
    for i in range(windows):
        chunk = directional[i * size: (i + 1) * size] if i < windows - 1 else directional[i * size:]
        wins = sum(1 for e in chunk if e.error_category in _CORRECT_DIRECTIONAL)
        out.append({
            "window": i,
            "start_ts": chunk[0].timestamp,
            "end_ts": chunk[-1].timestamp,
            "n": len(chunk),
            "precision": safe_div(wins, len(chunk)),
            "expectancy": mean([e.profit_or_loss for e in chunk]),
        })
    precisions = [w["precision"] for w in out]
    # Simple least-squares slope of precision against window index.
    n = len(precisions)
    mx = (n - 1) / 2.0
    my = mean(precisions)
    denom = sum((i - mx) ** 2 for i in range(n))
    slope = safe_div(sum((i - mx) * (p - my) for i, p in enumerate(precisions)), denom)
    return {
        "windows": out,
        "slope": slope,
        # A meaningfully negative slope over >=5 windows is real degradation.
        "degrading": slope < -0.01,
        "n": len(directional),
    }


def summarise(experiences: Sequence[Experience]) -> Dict[str, Any]:
    """One call that produces everything the dashboard needs."""
    return {
        "errors": error_breakdown(experiences),
        "regimes": regime_failures(experiences),
        "timeframes": timeframe_failures(experiences),
        "calibration": confidence_reliability(experiences),
        "patterns": recurring_patterns(experiences),
        "trend": accuracy_trend(experiences),
    }
