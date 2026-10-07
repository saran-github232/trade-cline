"""Candle-geometry and price-action features.

Every function here reads only candles that have already closed at the
computation timestamp.  The caller is responsible for truncating the series
(``MarketSeries.visible_at``); these functions never look forward.

Wick and body measurements are normalised by the candle's own range wherever
possible, so the features are scale-free and comparable across assets whose
absolute prices differ by orders of magnitude (a 60,000 BTC candle and a
1.10 EURUSD candle produce comparable numbers).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from ..types import Candle, MarketSeries
from ..utils.stats import clamp, mean, safe_div

__all__ = [
    "analyse_price_action",
    "detect_patterns",
    "candle_geometry",
    "is_inside_bar",
    "is_engulfing",
    "is_rejection",
    "range_expansion_ratio",
]


def _last(candles: Sequence[Candle]) -> Candle:
    return candles[-1]


def candle_geometry(candle: Candle) -> Dict[str, float]:
    """Body/wick decomposition of a single candle, normalised by its range.

    A zero-range candle (a gap or a synthetic artefact) yields zeros rather
    than a division by zero.
    """
    rng = candle.range
    if rng <= 0:
        return {
            "body_ratio": 0.0, "upper_wick_ratio": 0.0, "lower_wick_ratio": 0.0,
            "body_wick_ratio": 0.0, "close_position": 0.5, "direction": 0.0,
        }
    body = candle.body / rng
    upper = candle.upper_wick / rng
    lower = candle.lower_wick / rng
    # Signed body/wick skew: positive means the close rejected the lows.
    wick_total = candle.upper_wick + candle.lower_wick
    skew = safe_div(candle.lower_wick - candle.upper_wick, wick_total, 0.0) if wick_total > 0 else 0.0
    return {
        "body_ratio": body,
        "upper_wick_ratio": upper,
        "lower_wick_ratio": lower,
        "body_wick_ratio": skew,
        "close_position": clamp(safe_div(candle.close - candle.low, rng, 0.5)),
        "direction": 1.0 if candle.is_bullish else (-1.0 if candle.is_bearish else 0.0),
    }


def is_inside_bar(previous: Candle, current: Candle) -> bool:
    """True when the current candle's range sits inside the previous range."""
    return current.high <= previous.high and current.low >= previous.low


def is_engulfing(previous: Candle, current: Candle) -> Tuple[bool, bool]:
    """Return ``(bullish_engulfing, bearish_engulfing)``.

    Engulfing requires the *bodies* to overlap fully and the directions to
    oppose, which is stricter (and less noisy) than comparing wicks.
    """
    bull = (
        previous.is_bearish and current.is_bullish
        and current.close >= previous.open and current.open <= previous.close
    )
    bear = (
        previous.is_bullish and current.is_bearish
        and current.close <= previous.open and current.open >= previous.close
    )
    return (bull, bear)


def is_rejection(candle: Candle, *, min_wick_ratio: float = 0.55) -> Tuple[bool, bool]:
    """Return ``(bullish_rejection, bearish_rejection)``.

    A rejection candle has a dominant wick against a small body, i.e. price
    probed a level and was pushed back.  The threshold is deliberately high so
    that ordinary candles are not mislabelled as reversals.
    """
    rng = candle.range
    if rng <= 0:
        return (False, False)
    lower = candle.lower_wick / rng
    upper = candle.upper_wick / rng
    body = candle.body_abs / rng
    bull = lower >= min_wick_ratio and body <= 0.35 and candle.close >= candle.open
    bear = upper >= min_wick_ratio and body <= 0.35 and candle.close <= candle.open
    return (bull, bear)


def range_expansion_ratio(candles: Sequence[Candle], *, lookback: int = 20) -> float:
    """Current range divided by the average range of the preceding window.

    Values well above 1.0 indicate expansion (often a breakout or a spike);
    values below 1.0 indicate compression.
    """
    if len(candles) < 3:
        return 1.0
    window = candles[-(lookback + 1):-1] if len(candles) > lookback else candles[:-1]
    avg = mean([c.range for c in window])
    if not math.isfinite(avg) or avg <= 0:
        return 1.0
    return candles[-1].range / avg


def detect_patterns(series: MarketSeries) -> Dict[str, Any]:
    """Named boolean/float pattern flags for the most recent candles."""
    candles = series.candles
    out: Dict[str, Any] = {
        "inside_bar": False, "bullish_engulfing": False, "bearish_engulfing": False,
        "bullish_rejection": False, "bearish_rejection": False,
        "doji": False, "range_expansion": False, "range_expansion_ratio": 1.0,
        "bullish_sequence": 0, "bearish_sequence": 0,
    }
    if not candles:
        return out

    last = candles[-1]
    out["doji"] = last.is_doji
    out["range_expansion_ratio"] = round(range_expansion_ratio(candles), 4)
    out["range_expansion"] = out["range_expansion_ratio"] >= 1.5

    if len(candles) >= 2:
        prev = candles[-2]
        out["inside_bar"] = is_inside_bar(prev, last)
        bull, bear = is_engulfing(prev, last)
        out["bullish_engulfing"], out["bearish_engulfing"] = bull, bear

    bull_rej, bear_rej = is_rejection(last)
    out["bullish_rejection"], out["bearish_rejection"] = bull_rej, bear_rej

    # Consecutive same-direction closes, capped so one long run cannot dominate.
    streak = 0
    direction = 0
    for candle in reversed(candles):
        step = 1 if candle.is_bullish else (-1 if candle.is_bearish else 0)
        if step == 0 or (direction != 0 and step != direction):
            break
        direction = step
        streak += 1
        if streak >= 10:
            break
    if direction > 0:
        out["bullish_sequence"] = streak
    elif direction < 0:
        out["bearish_sequence"] = streak
    return out


def analyse_price_action(series: MarketSeries, *, lookback: int = 20) -> Dict[str, float]:
    """Full price-action feature block for the most recent candle.

    Feature names are prefixed ``pa.`` so they namespace cleanly with the
    trend/momentum/volatility/structure blocks.
    """
    candles = series.candles
    features: Dict[str, float] = {}
    if not candles:
        return features

    last = candles[-1]
    geometry = candle_geometry(last)
    features["pa.body_ratio"] = geometry["body_ratio"]
    features["pa.upper_wick_ratio"] = geometry["upper_wick_ratio"]
    features["pa.lower_wick_ratio"] = geometry["lower_wick_ratio"]
    features["pa.body_wick_ratio"] = geometry["body_wick_ratio"]
    features["pa.close_position"] = geometry["close_position"]
    features["pa.last_candle_direction"] = geometry["direction"]

    patterns = detect_patterns(series)
    features["pa.inside_bar"] = 1.0 if patterns["inside_bar"] else 0.0
    features["pa.bullish_engulfing"] = 1.0 if patterns["bullish_engulfing"] else 0.0
    features["pa.bearish_engulfing"] = 1.0 if patterns["bearish_engulfing"] else 0.0
    features["pa.bullish_rejection"] = 1.0 if patterns["bullish_rejection"] else 0.0
    features["pa.bearish_rejection"] = 1.0 if patterns["bearish_rejection"] else 0.0
    features["pa.doji"] = 1.0 if patterns["doji"] else 0.0
    features["pa.range_expansion"] = 1.0 if patterns["range_expansion"] else 0.0
    features["pa.range_expansion_ratio"] = patterns["range_expansion_ratio"]
    features["pa.bullish_sequence"] = float(patterns["bullish_sequence"])
    features["pa.bearish_sequence"] = float(patterns["bearish_sequence"])

    # Net directional pressure over the recent window: how many candles closed
    # in the upper half of their range versus the lower half.  This is a cheap,
    # robust proxy for who is in control.
    window = candles[-lookback:] if len(candles) >= lookback else candles
    upper_closes = sum(1 for c in window if c.range > 0 and (c.close - c.low) / c.range > 0.6)
    lower_closes = sum(1 for c in window if c.range > 0 and (c.close - c.low) / c.range < 0.4)
    features["pa.upper_close_share"] = safe_div(upper_closes, len(window), 0.5)
    features["pa.lower_close_share"] = safe_div(lower_closes, len(window), 0.5)
    features["pa.pressure"] = features["pa.upper_close_share"] - features["pa.lower_close_share"]

    # Mean body size relative to mean range: high values mean decisive candles.
    bodies = [c.body_abs for c in window]
    ranges = [c.range for c in window]
    features["pa.mean_body_ratio"] = safe_div(mean(bodies), mean(ranges), 0.0)
    return features
