"""Deterministic, dependency-free chart-vision analyser.

This is the DEFAULT provider and must always work: no network, no API key, no
third-party library.  Every number it reports is derived from the inputs it is
handed - the OHLC ``series`` and the structured numerical ``summary`` - and it
never invents a price level that is not observable in the data.

It deliberately re-implements a *small* amount of price-action logic (local
swing detection, moving averages, volatility) rather than importing
``market_ai.features``: the feature module is owned by another subsystem and
keeping the vision path self-contained guarantees the fallback is always
available and never has a hidden dependency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..types import Direction, MarketSeries, Timeframe
from ..utils.stats import clamp, stdev
from .schema import VisionResult

__all__ = ["OfflineChartAnalyser"]


# --------------------------------------------------------------------------
# small, local, deterministic numerical helpers
# --------------------------------------------------------------------------
def _sma(values: Sequence[float], period: int) -> float:
    """Simple moving average of the last ``period`` values (NaN when empty)."""
    if not values:
        return float("nan")
    period = max(1, min(period, len(values)))
    window = values[-period:]
    return sum(window) / len(window)


def _norm_slope(values: Sequence[float], period: int) -> float:
    """Least-squares slope of the last ``period`` values, normalised by price.

    Normalising by the mean price makes the slope scale-free so a threshold can
    be compared across instruments quoted at wildly different levels.
    """
    vals = list(values[-period:]) if period > 0 else []
    m = len(vals)
    if m < 2:
        return 0.0
    xbar = (m - 1) / 2.0
    ybar = sum(vals) / m
    den = sum((x - xbar) ** 2 for x in range(m))
    if den == 0:
        return 0.0
    num = sum((x - xbar) * (y - ybar) for x, y in enumerate(vals))
    scale = abs(ybar) if abs(ybar) > 1e-12 else 1.0
    return (num / den) / scale


def _find_swings(
    highs: Sequence[float], lows: Sequence[float], k: int = 2
) -> Tuple[List[int], List[int]]:
    """Return indices of local swing highs and swing lows.

    A bar is a swing high when its high is the maximum of a ``2k+1`` window
    centred on it (and strictly above the window minimum, to exclude flat
    plateaus).  Swing lows are the mirror image.  This is a deliberately simple
    local-extremum scan: deterministic, dependency-free and cheap.
    """
    swing_highs: List[int] = []
    swing_lows: List[int] = []
    n = len(highs)
    if n < 2 * k + 1:
        return swing_highs, swing_lows
    for i in range(k, n - k):
        h_win = highs[i - k : i + k + 1]
        l_win = lows[i - k : i + k + 1]
        if highs[i] == max(h_win) and highs[i] > min(h_win):
            swing_highs.append(i)
        if lows[i] == min(l_win) and lows[i] < max(l_win):
            swing_lows.append(i)
    return swing_highs, swing_lows


def _cluster_levels(levels: Sequence[float], tolerance: float) -> List[float]:
    """Collapse near-identical price levels into their means."""
    cleaned = sorted(float(x) for x in levels if math.isfinite(float(x)))
    if not cleaned:
        return []
    clusters: List[List[float]] = [[cleaned[0]]]
    for value in cleaned[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(c) / len(c) for c in clusters]


def _pct_returns(closes: Sequence[float]) -> List[float]:
    """Simple percentage returns between consecutive closes."""
    out: List[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev and math.isfinite(prev) and prev != 0:
            out.append((cur - prev) / prev)
    return out


# --------------------------------------------------------------------------
# analyser
# --------------------------------------------------------------------------
class OfflineChartAnalyser:
    """Deterministic chart analysis built purely from the supplied inputs.

    The result is a :class:`VisionResult` with ``provider="offline"`` and
    ``degraded=False``.  It is always available and is the fallback whenever an
    external provider fails.
    """

    #: Directional thresholds.  Small and explicit so behaviour is auditable.
    _TREND_EPS = 0.001        # relative fast/slow SMA separation to call a trend
    _SLOPE_EPS = 2e-4         # normalised slope threshold
    _MOMENTUM_EPS = 0.002     # relative rate-of-change threshold
    _VOL_HIGH = 1.3           # recent/full stdev ratio above which vol is "high"
    _VOL_LOW = 0.7            # ... below which vol is "low"

    def __init__(self, config: Optional[Config] = None) -> None:
        """Store configuration; defaults are safe and offline-only."""
        self.config = config or Config()
        self.name = "offline"

    def analyse(
        self,
        *,
        series: MarketSeries,
        summary: Mapping[str, Any],
        timeframe: Optional[Timeframe] = None,
        asset: Optional[str] = None,
    ) -> VisionResult:
        """Analyse ``series`` (optionally informed by ``summary``).

        Returns a well-formed :class:`VisionResult`.  Degenerate inputs (empty
        or single-bar series) yield a neutral, low-confidence result rather
        than an exception, because the vision layer must never crash the
        pipeline.
        """
        summary = summary if isinstance(summary, Mapping) else {}
        closes = list(series.closes) if series is not None else []
        highs = list(series.highs) if series is not None else []
        lows = list(series.lows) if series is not None else []

        if len(closes) < 2:
            return self._degenerate(closes, summary)

        price = closes[-1]
        hi_range = max(highs)
        lo_range = min(lows)

        trend, slope = self._classify_trend(closes, summary)
        structure = self._classify_structure(highs, lows)
        support, resistance = self._support_resistance(
            highs, lows, price, hi_range, lo_range
        )
        momentum = self._classify_momentum(closes, summary)
        volatility = self._classify_volatility(closes)
        direction = self._direction_bias(trend, slope, momentum)
        breakout = self._breakout_probability(
            price, support, resistance, hi_range, lo_range, momentum, slope
        )
        reversal = self._reversal_probability(
            price, support, resistance, hi_range, lo_range, direction, slope, momentum
        )
        confidence = self._confidence(closes, trend, slope, volatility)
        invalidating = self._invalidating_conditions(
            direction, support, resistance, volatility
        )
        reasoning = self._reasoning_summary(
            trend, structure, momentum, volatility, direction, confidence
        )

        raw = {
            "source": "offline",
            "trend": trend,
            "structure": structure,
            "momentum": momentum,
            "volatility": volatility,
            "direction_bias": direction.value,
            "support": list(support),
            "resistance": list(resistance),
            "breakout_probability": breakout,
            "reversal_probability": reversal,
            "confidence": confidence,
            "bars": len(closes),
            "last_price": price,
        }

        return VisionResult(
            direction_bias=direction,
            trend=trend,
            structure=structure,
            support=support,
            resistance=resistance,
            breakout_probability=breakout,
            reversal_probability=reversal,
            momentum=momentum,
            volatility=volatility,
            confidence=confidence,
            invalidating_conditions=invalidating,
            reasoning_summary=reasoning,
            provider="offline",
            degraded=False,
            raw=raw,
        )


    # -- classification helpers ------------------------------------------
    @staticmethod
    def _summary_number(summary: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
        """Return the first finite numeric value found under ``keys``."""
        for key in keys:
            if key in summary:
                val = summary[key]
                if (
                    isinstance(val, (int, float))
                    and not isinstance(val, bool)
                    and math.isfinite(float(val))
                ):
                    return float(val)
        return None

    def _classify_trend(
        self, closes: Sequence[float], summary: Mapping[str, Any]
    ) -> Tuple[str, float]:
        """Return ``(trend_label, normalised_slope)`` from fast/slow SMA + slope."""
        fast = _sma(closes, 20)
        slow = _sma(closes, 50)
        slope = _norm_slope(closes, min(20, len(closes)))
        sep = 0.0
        if math.isfinite(slow) and slow != 0 and math.isfinite(fast):
            sep = (fast - slow) / slow
        if sep > self._TREND_EPS and slope > self._SLOPE_EPS:
            return "up", slope
        if sep < -self._TREND_EPS and slope < -self._SLOPE_EPS:
            return "down", slope
        return "sideways", slope

    def _classify_structure(
        self, highs: Sequence[float], lows: Sequence[float]
    ) -> str:
        """Label the swing structure as trending, ranging or mixed."""
        swing_highs, swing_lows = _find_swings(highs, lows, 2)
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = highs[swing_highs[-1]] > highs[swing_highs[-2]]
            hl = lows[swing_lows[-1]] > lows[swing_lows[-2]]
            lh = highs[swing_highs[-1]] < highs[swing_highs[-2]]
            ll = lows[swing_lows[-1]] < lows[swing_lows[-2]]
            if hh and hl:
                return "higher_highs_higher_lows"
            if lh and ll:
                return "lower_highs_lower_lows"
            if not hh and not lh and not hl and not ll:
                return "range"
            return "mixed"
        if len(swing_highs) < 2 and len(swing_lows) < 2:
            return "range"
        return "mixed"

    def _support_resistance(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        price: float,
        hi_range: float,
        lo_range: float,
    ) -> Tuple[List[float], List[float]]:
        """Derive support/resistance levels from swing extremes.

        Every returned level is either a real swing price or the recent
        window extreme, and all are clamped into the observed ``[lo, hi]``
        range, so no level can ever be invented outside the data.
        """
        swing_highs, swing_lows = _find_swings(highs, lows, 2)
        span = max(hi_range - lo_range, 0.0)
        tolerance = max(span * 0.01, abs(price) * 0.001, 1e-9)

        sup_levels = _cluster_levels([lows[i] for i in swing_lows], tolerance)
        res_levels = _cluster_levels([highs[i] for i in swing_highs], tolerance)

        supports = sorted([s for s in sup_levels if s <= price], reverse=True)[:3]
        resistances = sorted([r for r in res_levels if r >= price])[:3]

        lookback = min(50, len(lows))
        if not supports:
            supports = [min(min(lows[-lookback:]), price)]
        if not resistances:
            resistances = [max(max(highs[-lookback:]), price)]

        supports = [clamp(s, lo_range, hi_range) for s in supports]
        resistances = [clamp(r, lo_range, hi_range) for r in resistances]
        return supports, resistances

    def _classify_momentum(
        self, closes: Sequence[float], summary: Mapping[str, Any]
    ) -> str:
        """Label momentum from the rate of change (with an optional RSI hint)."""
        lookback = min(14, len(closes) - 1)
        if lookback < 1:
            return "flat"
        base = closes[-1 - lookback]
        roc = (closes[-1] - base) / base if base else 0.0
        thr = self._MOMENTUM_EPS
        if roc > 3 * thr:
            return "strong_up"
        if roc > thr:
            return "up"
        if roc < -3 * thr:
            return "strong_down"
        if roc < -thr:
            return "down"
        # Flat price action: fall back to a supplied RSI reading when present.
        rsi = self._summary_number(summary, ("rsi", "rsi_14", "momentum.rsi_14"))
        if rsi is not None:
            if rsi > 65:
                return "up"
            if rsi < 35:
                return "down"
        return "flat"

    def _classify_volatility(self, closes: Sequence[float]) -> str:
        """Label volatility by recent vs full-sample return dispersion."""
        returns = _pct_returns(closes)
        if len(returns) < 4:
            return "normal"
        recent = stdev(returns[-min(20, len(returns)) :])
        full = stdev(returns)
        if not math.isfinite(full) or full <= 0 or not math.isfinite(recent):
            return "normal"
        ratio = recent / full
        if ratio > self._VOL_HIGH:
            return "high"
        if ratio < self._VOL_LOW:
            return "low"
        return "normal"



    def _direction_bias(
        self, trend: str, slope: float, momentum: str
    ) -> Direction:
        """Combine trend and momentum into a directional bias."""
        if trend == "up":
            return Direction.UP
        if trend == "down":
            return Direction.DOWN
        # Sideways: only a *strong* momentum reading may break the tie.
        if momentum == "strong_up":
            return Direction.UP
        if momentum == "strong_down":
            return Direction.DOWN
        return Direction.NEUTRAL

    def _breakout_probability(
        self,
        price: float,
        support: Sequence[float],
        resistance: Sequence[float],
        hi_range: float,
        lo_range: float,
        momentum: str,
        slope: float,
    ) -> float:
        """Estimate the probability of an upside breakout, in ``[0, 1]``."""
        span = hi_range - lo_range
        if span <= 0:
            return 0.0
        near_res = 0.0
        if resistance:
            near_res = clamp(1.0 - abs(resistance[0] - price) / span, 0.0, 1.0)
        mom = {"strong_up": 1.0, "up": 0.7, "flat": 0.4, "down": 0.3, "strong_down": 0.0}
        align = 1.0 if slope > 0 else (0.0 if slope < 0 else 0.5)
        prob = 0.5 * near_res + 0.3 * mom.get(momentum, 0.4) + 0.2 * align
        return clamp(prob, 0.0, 1.0)

    def _reversal_probability(
        self,
        price: float,
        support: Sequence[float],
        resistance: Sequence[float],
        hi_range: float,
        lo_range: float,
        direction: Direction,
        slope: float,
        momentum: str,
    ) -> float:
        """Estimate the probability of a reversal, in ``[0, 1]``."""
        span = hi_range - lo_range
        if span <= 0:
            return 0.0
        stretch = 0.0
        if direction == Direction.UP and resistance:
            stretch = clamp(1.0 - abs(resistance[0] - price) / span, 0.0, 1.0)
        elif direction == Direction.DOWN and support:
            stretch = clamp(1.0 - abs(price - support[0]) / span, 0.0, 1.0)
        weakening = (
            (slope > 0 and momentum in ("down", "strong_down"))
            or (slope < 0 and momentum in ("up", "strong_up"))
        )
        prob = 0.6 * stretch + (0.3 if weakening else 0.0)
        return clamp(prob, 0.0, 1.0)

    def _confidence(
        self,
        closes: Sequence[float],
        trend: str,
        slope: float,
        volatility: str,
    ) -> float:
        """Self-reported confidence, capped by ``max_self_reported_confidence``."""
        clarity = clamp(abs(slope) / (self._SLOPE_EPS * 10.0), 0.0, 1.0)
        trend_bonus = 0.25 if trend in ("up", "down") else 0.0
        vol_penalty = 0.15 if volatility == "high" else 0.0
        conf = clamp(0.35 + 0.3 * clarity + trend_bonus - vol_penalty, 0.0, 1.0)
        cap = float(self.config.vision.max_self_reported_confidence)
        return min(conf, cap) if math.isfinite(cap) else conf

    def _invalidating_conditions(
        self,
        direction: Direction,
        support: Sequence[float],
        resistance: Sequence[float],
        volatility: str,
    ) -> List[str]:
        """Plain-language conditions that would invalidate this read."""
        out: List[str] = []
        if direction == Direction.UP and support:
            out.append(
                f"A close below support {support[0]:.6g} would invalidate the bullish bias."
            )
        elif direction == Direction.DOWN and resistance:
            out.append(
                f"A close above resistance {resistance[0]:.6g} would invalidate the bearish bias."
            )
        else:
            if support:
                out.append(f"A break below {support[0]:.6g} would favour the downside.")
            if resistance:
                out.append(f"A break above {resistance[0]:.6g} would favour the upside.")
        if volatility == "high":
            out.append(
                "Sustained volatility beyond the recent range would invalidate the "
                "current structure."
            )
        out.append("Stale or incomplete data would invalidate this read.")
        return out

    def _reasoning_summary(
        self,
        trend: str,
        structure: str,
        momentum: str,
        volatility: str,
        direction: Direction,
        confidence: float,
    ) -> str:
        """Compose a short, human-readable summary of the analysis."""
        return (
            f"Trend {trend} with {structure} structure; momentum {momentum}; "
            f"volatility {volatility}. Directional bias {direction.value} at "
            f"confidence {confidence:.2f}."
        )

    def _degenerate(
        self, closes: Sequence[float], summary: Mapping[str, Any]
    ) -> VisionResult:
        """Return a neutral, low-confidence result for insufficient data."""
        return VisionResult(
            direction_bias=Direction.NEUTRAL,
            trend="unknown",
            structure="unknown",
            support=[],
            resistance=[],
            breakout_probability=0.0,
            reversal_probability=0.0,
            momentum="unknown",
            volatility="unknown",
            confidence=0.0,
            invalidating_conditions=["Insufficient data to analyse the chart."],
            reasoning_summary=(
                "Insufficient price history to form a view; defaulting to neutral."
            ),
            provider="offline",
            degraded=False,
            raw={"source": "offline", "bars": len(closes), "note": "insufficient-data"},
        )

