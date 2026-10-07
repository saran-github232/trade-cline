"""Market-structure analysis: swings, trends, support/resistance, breakouts.

A swing high is only *confirmed* once ``lookback`` candles have printed on
each side of it.  That confirmation delay is a real cost of being causal: the
most recent ``lookback`` candles can never contain a confirmed swing.  This is
intentional - calling a swing in real time is exactly the look-ahead error
this module is designed to avoid.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import Candle, MarketSeries
from ..utils.stats import clamp, mean, safe_div

__all__ = [
    "find_swings",
    "analyse_structure",
    "support_resistance_zones",
    "swing_labels",
    "detect_breakouts",
]


def find_swings(
    highs: Sequence[float], lows: Sequence[float], lookback: int = 3
) -> Tuple[List[int], List[int]]:
    """Return ``(swing_high_indices, swing_low_indices)``.

    An index ``i`` is a swing high when ``highs[i]`` is strictly greater than
    every high in the ``lookback`` bars on either side.  Only indices where the
    full window exists are considered, which is what makes the result causal:
    a swing at index ``i`` is knowable at bar ``i + lookback``.
    """
    swing_highs: List[int] = []
    swing_lows: List[int] = []
    n = min(len(highs), len(lows))
    if lookback < 1 or n < 2 * lookback + 1:
        return swing_highs, swing_lows
    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback:i + lookback + 1]
        if highs[i] == max(window_high) and window_high.count(highs[i]) == 1:
            swing_highs.append(i)
        window_low = lows[i - lookback:i + lookback + 1]
        if lows[i] == min(window_low) and window_low.count(lows[i]) == 1:
            swing_lows.append(i)
    return swing_highs, swing_lows


def swing_labels(
    highs: Sequence[float], lows: Sequence[float], lookback: int = 3
) -> Dict[str, Any]:
    """Classify the last two swings into HH/HL/LH/LL structure."""
    sh, sl = find_swings(highs, lows, lookback)
    out: Dict[str, Any] = {
        "swing_highs": sh, "swing_lows": sl,
        "higher_high": 0.0, "higher_low": 0.0,
        "lower_high": 0.0, "lower_low": 0.0,
        "last_swing_high": None, "last_swing_low": None,
    }
    if len(sh) >= 2:
        out["last_swing_high"] = highs[sh[-1]]
        if highs[sh[-1]] > highs[sh[-2]]:
            out["higher_high"] = 1.0
        elif highs[sh[-1]] < highs[sh[-2]]:
            out["lower_high"] = 1.0
    if len(sl) >= 2:
        out["last_swing_low"] = lows[sl[-1]]
        if lows[sl[-1]] > lows[sl[-2]]:
            out["higher_low"] = 1.0
        elif lows[sl[-1]] < lows[sl[-2]]:
            out["lower_low"] = 1.0
    return out


def support_resistance_zones(
    series: MarketSeries, *, lookback: int = 120, tolerance: float = 0.002
) -> Dict[str, Any]:
    """Cluster swing levels into support and resistance zones.

    Levels within ``tolerance`` (a fraction of price) of each other are merged
    and weighted by how many times price reacted there, so a level touched five
    times outranks a one-off spike.
    """
    candles = series.candles[-lookback:] if len(series.candles) > lookback else series.candles
    if len(candles) < 10:
        return {"support": [], "resistance": [], "nearest_support": None, "nearest_resistance": None}

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    sh, sl = find_swings(highs, lows, lookback=3)
    price = candles[-1].close

    def cluster(indices: Sequence[int], source: Sequence[float]) -> List[Dict[str, float]]:
        levels = sorted(source[i] for i in indices)
        zones: List[Dict[str, float]] = []
        for level in levels:
            if zones and abs(level - zones[-1]["level"]) <= tolerance * max(price, 1e-9):
                zone = zones[-1]
                zone["level"] = (zone["level"] * zone["touches"] + level) / (zone["touches"] + 1)
                zone["touches"] += 1
            else:
                zones.append({"level": level, "touches": 1})
        return sorted(zones, key=lambda z: (-z["touches"], z["level"]))

    resistance_zones = [z for z in cluster(sh, highs) if z["level"] > price]
    support_zones = [z for z in cluster(sl, lows) if z["level"] < price]
    resistance_zones.sort(key=lambda z: z["level"])
    support_zones.sort(key=lambda z: -z["level"])

    return {
        "support": support_zones[:5],
        "resistance": resistance_zones[:5],
        "nearest_support": support_zones[0]["level"] if support_zones else None,
        "nearest_resistance": resistance_zones[0]["level"] if resistance_zones else None,
    }


def detect_breakouts(
    series: MarketSeries, *, lookback: int = 20, confirm_ratio: float = 0.001
) -> Dict[str, float]:
    """Detect breakouts *and* failed breakouts against the prior range.

    The distinction matters: a breakout that immediately fails is a strong
    counter-signal, and the structure signal provider scores it accordingly.

    The comparison window excludes the current bar, so the range being broken
    was genuinely knowable before the break.
    """
    candles = series.candles
    out = {
        "breakout_up": 0.0, "breakout_down": 0.0,
        "failed_breakout_up": 0.0, "failed_breakout_down": 0.0,
        "prior_range_high": float("nan"), "prior_range_low": float("nan"),
    }
    if len(candles) < lookback + 2:
        return out

    window = candles[-(lookback + 1):-1]
    prior_high = max(c.high for c in window)
    prior_low = min(c.low for c in window)
    out["prior_range_high"] = prior_high
    out["prior_range_low"] = prior_low

    last = candles[-1]
    buffer = confirm_ratio * max(last.close, 1e-9)

    if last.close > prior_high + buffer:
        out["breakout_up"] = 1.0
    elif last.high > prior_high + buffer and last.close <= prior_high:
        # Price traded above the range but closed back inside: failure.
        out["failed_breakout_up"] = 1.0

    if last.close < prior_low - buffer:
        out["breakout_down"] = 1.0
    elif last.low < prior_low - buffer and last.close >= prior_low:
        out["failed_breakout_down"] = 1.0
    return out


def analyse_structure(series: MarketSeries, *, lookback: int = 60) -> Dict[str, float]:
    """Full market-structure feature block.

    Distances to support/resistance are expressed in ATR units, so "1.5 ATR
    below resistance" means the same thing on every asset.
    """
    features: Dict[str, float] = {}
    candles = series.candles
    if len(candles) < 10:
        return features

    highs = series.highs
    lows = series.lows
    labels = swing_labels(highs, lows, lookback=3)
    features["structure.higher_high"] = labels["higher_high"]
    features["structure.higher_low"] = labels["higher_low"]
    features["structure.lower_high"] = labels["lower_high"]
    features["structure.lower_low"] = labels["lower_low"]

    bull_structure = labels["higher_high"] + labels["higher_low"]
    bear_structure = labels["lower_high"] + labels["lower_low"]
    features["structure.trend_score"] = (bull_structure - bear_structure) / 2.0

    window = candles[-lookback:] if len(candles) > lookback else candles
    swing_highs, swing_lows = find_swings(
        [c.high for c in window], [c.low for c in window], lookback=3
    )
    features["structure.swing_high_count"] = float(len(swing_highs))
    features["structure.swing_low_count"] = float(len(swing_lows))

    breakouts = detect_breakouts(series, lookback=20)
    features["structure.breakout_up"] = breakouts["breakout_up"]
    features["structure.breakout_down"] = breakouts["breakout_down"]
    features["structure.failed_breakout_up"] = breakouts["failed_breakout_up"]
    features["structure.failed_breakout_down"] = breakouts["failed_breakout_down"]

    zones = support_resistance_zones(series)
    last_close = candles[-1].close

    # ATR is used purely as a unit of distance here; it is computed inline to
    # keep this module independent of the indicator library's import order.
    trs: List[float] = []
    for i in range(1, min(len(candles), 15)):
        candle = candles[-i]
        prev_close = candles[-i - 1].close
        trs.append(max(candle.high - candle.low,
                       abs(candle.high - prev_close),
                       abs(candle.low - prev_close)))
    atr = mean(trs) if trs else 0.0

    if atr and atr > 0:
        if zones["nearest_resistance"] is not None:
            features["structure.distance_to_resistance_atr"] = (
                zones["nearest_resistance"] - last_close
            ) / atr
        else:
            features["structure.distance_to_resistance_atr"] = 5.0
        if zones["nearest_support"] is not None:
            features["structure.distance_to_support_atr"] = (
                last_close - zones["nearest_support"]
            ) / atr
        else:
            features["structure.distance_to_support_atr"] = 5.0

    # Position inside the recent range: 1.0 at the highs, 0.0 at the lows.
    recent = candles[-lookback:] if len(candles) > lookback else candles
    range_high = max(c.high for c in recent)
    range_low = min(c.low for c in recent)
    span = range_high - range_low
    features["structure.range_position"] = clamp(safe_div(last_close - range_low, span, 0.5))
    features["structure.range_span_atr"] = safe_div(span, atr, 0.0) if atr > 0 else 0.0
    return features
