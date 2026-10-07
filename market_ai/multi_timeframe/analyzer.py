"""Hierarchical multi-timeframe analysis.

The hierarchy is fixed and one-directional::

    higher timeframe  ->  broader market structure (the bias)
    middle timeframe  ->  the setup
    lower timeframe   ->  the trigger / confirmation

The lower timeframe may *strengthen or weaken* the higher-timeframe bias, but
it may never invert it.  A 1m reversal inside a strong 15m uptrend is a
pullback, not a trend change, and treating it as one is a classic way to lose
money on a chart that was actually fine.  When the timeframes genuinely
disagree the correct output is a higher ``conflict`` score, which raises the
NO-TRADE probability in the ensemble.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..types import Direction, MarketSeries, Timeframe
from ..utils.logging import get_logger
from ..utils.stats import clamp, mean, safe_div

__all__ = ["TimeframeView", "MultiTimeframeResult", "MultiTimeframeAnalyzer"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class TimeframeView:
    """A single timeframe's directional read."""

    timeframe: Timeframe
    direction: Direction
    strength: float
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeframe": self.timeframe.value,
            "direction": self.direction.value,
            "strength": round(float(self.strength), 6),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class MultiTimeframeResult:
    """Combined hierarchical view."""

    higher: Optional[TimeframeView]
    middle: Optional[TimeframeView]
    lower: Optional[TimeframeView]
    agreement: float
    conflict: float
    aligned_direction: Direction
    timestamp: int
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "higher": self.higher.to_dict() if self.higher else None,
            "middle": self.middle.to_dict() if self.middle else None,
            "lower": self.lower.to_dict() if self.lower else None,
            "agreement": round(float(self.agreement), 6),
            "conflict": round(float(self.conflict), 6),
            "aligned_direction": self.aligned_direction.value,
            "timestamp": int(self.timestamp),
            "notes": list(self.notes),
        }


def _slope_score(closes: Sequence[float], window: int = 20) -> float:
    """Normalised least-squares slope of the trailing window.

    Returns a scale-free number: a value of 1.0 means the trend moves roughly
    one standard deviation of price per bar, which is a strong, sustained move.
    """
    values = [v for v in closes[-window:] if v == v]
    n = len(values)
    if n < 4:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(values) / n
    denom = sum((i - mx) ** 2 for i in range(n))
    if denom <= 0:
        return 0.0
    slope = sum((i - mx) * (v - my) for i, v in enumerate(values)) / denom
    scale = abs(my) if abs(my) > 1e-12 else 1.0
    return (slope / scale) * 100.0


class MultiTimeframeAnalyzer:
    """Builds a higher/middle/lower view and measures their alignment."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    # -- single timeframe --------------------------------------------------
    def analyse_timeframe(self, series: MarketSeries) -> TimeframeView:
        """Directional read for one timeframe, using only visible candles."""
        closes = series.closes
        tf = series.timeframe
        if len(closes) < 12:
            return TimeframeView(tf, Direction.NEUTRAL, 0.0, {"reason": "insufficient bars"})

        slope = _slope_score(closes)
        window = closes[-20:] if len(closes) >= 20 else closes
        net = safe_div(window[-1] - window[0], window[0], 0.0)
        momentum = safe_div(net, abs(slope) / 100.0 + 1e-9, 0.0) if slope else 0.0

        # Composite strength combines trend slope with net displacement.
        strength_raw = slope * 8.0 + net * 40.0
        strength = clamp(abs(strength_raw) / 3.0)

        if strength_raw > 0.35:
            direction = Direction.UP
        elif strength_raw < -0.35:
            direction = Direction.DOWN
        else:
            direction = Direction.NEUTRAL
            strength *= 0.5  # a neutral read is weak evidence either way

        detail = {
            "slope": round(slope, 6),
            "net_change": round(net, 6),
            "bars": len(series),
            "last_close": closes[-1],
        }
        return TimeframeView(tf, direction, clamp(strength), detail)

    # -- hierarchy ---------------------------------------------------------
    def analyse(
        self,
        views: Mapping[Timeframe, MarketSeries],
        as_of: int,
        decision_timeframe: Timeframe,
    ) -> MultiTimeframeResult:
        """Analyse all available timeframes relative to the decision timeframe.

        Timeframes slower than the decision timeframe are "higher"; faster ones
        are "lower".  The nearest available on each side is used, which keeps
        the hierarchy meaningful even when, say, only 5m and 15m exist.
        """
        decision_timeframe = Timeframe.parse(decision_timeframe)
        ranked = sorted(
            (tf for tf, s in views.items() if len(s) > 0),
            key=lambda t: t.rank,
        )
        if not ranked:
            return MultiTimeframeResult(None, None, None, 0.0, 1.0, Direction.NEUTRAL, as_of,
                                        ("no timeframe data",))

        decision_rank = decision_timeframe.rank
        higher_tfs = [tf for tf in ranked if tf.rank > decision_rank]
        lower_tfs = [tf for tf in ranked if tf.rank < decision_rank]

        middle_tf = decision_timeframe if decision_timeframe in ranked else min(
            ranked, key=lambda t: abs(t.rank - decision_rank)
        )
        higher_tf = max(higher_tfs) if higher_tfs else None
        lower_tf = min(lower_tfs) if lower_tfs else None

        higher_view = self.analyse_timeframe(views[higher_tf]) if higher_tf else None
        middle_view = self.analyse_timeframe(views[middle_tf])
        lower_view = self.analyse_timeframe(views[lower_tf]) if lower_tf else None

        notes: List[str] = []
        available = [v for v in (higher_view, middle_view, lower_view) if v is not None]
        directional = [v for v in available if v.direction is not Direction.NEUTRAL]

        if not directional:
            return MultiTimeframeResult(
                higher_view, middle_view, lower_view, 0.0, 1.0, Direction.NEUTRAL, as_of,
                ("all timeframes neutral",),
            )

        ups = [v for v in directional if v.direction is Direction.UP]
        downs = [v for v in directional if v.direction is Direction.DOWN]

        # Weight each timeframe by how strongly it reads, so a strong 15m view
        # outranks a marginal 1m view rather than being outvoted by it.
        up_weight = sum(v.strength + 0.25 for v in ups)
        down_weight = sum(v.strength + 0.25 for v in downs)
        total_weight = up_weight + down_weight
        if total_weight <= 0:
            agreement = 0.0
            conflict = 1.0
        else:
            agreement = max(up_weight, down_weight) / total_weight
            conflict = min(up_weight, down_weight) / total_weight

        # The higher timeframe sets the bias.  If it is neutral, the middle
        # timeframe decides; only then does the lower timeframe get a say.
        bias_view = higher_view if (higher_view and higher_view.direction is not Direction.NEUTRAL) else middle_view
        if bias_view.direction is Direction.NEUTRAL and lower_view is not None:
            bias_view = lower_view
        aligned = bias_view.direction

        # Explicit conflict cases worth reporting, because they are exactly
        # where a naive system would take a bad trade.
        if higher_view and lower_view:
            if higher_view.direction is Direction.UP and lower_view.direction is Direction.DOWN:
                notes.append("lower timeframe contradicts a bullish higher timeframe (pullback, not reversal)")
            elif higher_view.direction is Direction.DOWN and lower_view.direction is Direction.UP:
                notes.append("lower timeframe contradicts a bearish higher timeframe (bounce, not reversal)")
        if higher_view and middle_view and higher_view.direction is not Direction.NEUTRAL \
                and middle_view.direction is not Direction.NEUTRAL \
                and higher_view.direction is not middle_view.direction:
            notes.append("middle timeframe disagrees with the higher timeframe")
        if higher_view is None:
            notes.append("no higher timeframe available; hierarchy is incomplete")

        # Conflict measured over all available views is more honest than over
        # directional ones only: an all-neutral panel is genuinely conflicted.
        if len(available) >= 2:
            counts: Dict[Direction, int] = {}
            for view in available:
                counts[view.direction] = counts.get(view.direction, 0) + 1
            conflict = 1.0 - max(counts.values()) / len(available)
            agreement = 1.0 - conflict

        return MultiTimeframeResult(
            higher_view, middle_view, lower_view,
            clamp(agreement), clamp(conflict), aligned, as_of, tuple(notes),
        )
