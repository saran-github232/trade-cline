"""Market-regime classification.

Why this matters more than another model
----------------------------------------
A directional model trained across all conditions averages over incompatible
market behaviours.  Splitting evaluation by regime is how the system discovers
that it is, say, profitable in trends and reliably wrong in ranges - and the
regime signal is what lets the gate refuse to trade in the conditions where
its own history says it has no edge.

When the evidence does not clearly support a label, the detector returns
``UNCERTAIN`` with low confidence rather than forcing a best guess.  The
decision gate treats UNCERTAIN as an automatic NO-TRADE, so an honest
"I don't know" here directly protects capital downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..config import Config, RegimeConfig
from ..types import FeatureVector, MarketSeries, Regime
from ..utils.logging import get_logger
from ..utils.stats import clamp, mean, percentile, safe_div

__all__ = ["RegimeResult", "RegimeDetector"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class RegimeResult:
    """A regime label with the evidence that produced it."""

    regime: Regime
    confidence: float
    supporting_features: Mapping[str, float] = field(default_factory=dict)
    timestamp: int = 0
    alternatives: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": round(float(self.confidence), 6),
            "supporting_features": {k: (round(v, 6) if isinstance(v, float) and v == v else v)
                                    for k, v in self.supporting_features.items()},
            "timestamp": int(self.timestamp),
            "alternatives": {k: round(v, 6) for k, v in self.alternatives.items()},
        }


def _safe(value: Any, default: float = float("nan")) -> float:
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return fval if math.isfinite(fval) else default


class RegimeDetector:
    """Rule-based, explainable regime classifier.

    Deliberately *not* a learned model.  A regime label drives NO-TRADE
    decisions and per-regime performance attribution; it must be inspectable
    and stable.  A learned regime model that silently drifts would make every
    historical regime report meaningless.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.settings: RegimeConfig = self.config.regime

    # -- feature extraction ------------------------------------------------
    def _compute(
        self, series: MarketSeries, features: Optional[FeatureVector]
    ) -> Dict[str, float]:
        """Gather the indicators the classifier needs.

        Prefers values already present on ``features`` (so the feature engine is
        the single source of truth) and only recomputes what is missing.
        """
        values: Dict[str, float] = {}
        if features is not None:
            for key in (
                "momentum.adx_14", "trend.ema_9", "trend.ema_21", "trend.ema_50",
                "momentum.rsi_14", "volatility.atr_14", "volatility.atr_percentile",
                "volatility.bb_width", "structure.trend_score",
                "structure.range_span_atr", "price.close",
            ):
                value = features.get(key)
                if value == value:  # not NaN
                    values[key] = float(value)

        closes, highs, lows = series.closes, series.highs, series.lows
        try:
            from ..features.indicators import adx, atr, ema, rolling_volatility
            if "momentum.adx_14" not in values and len(closes) > 30:
                values["momentum.adx_14"] = adx(highs, lows, closes, 14)["adx"][-1]
            if "volatility.atr_14" not in values and len(closes) > 20:
                values["volatility.atr_14"] = atr(highs, lows, closes, 14)[-1]
            if len(closes) > 60:
                if "trend.ema_21" not in values:
                    values["trend.ema_21"] = ema(closes, 21)[-1]
                if "trend.ema_50" not in values:
                    values["trend.ema_50"] = ema(closes, 50)[-1]
                values.setdefault("trend.ema_slope", safe_div(
                    ema(closes, 50)[-1] - ema(closes, 50)[-min(11, len(closes))],
                    abs(values.get("volatility.atr_14", 1.0)) or 1.0, 0.0))
        except Exception as exc:  # pragma: no cover - indicator layer missing
            _log.debug("indicator fallback in regime detector: %s", exc)

        if "price.close" not in values and closes:
            values["price.close"] = closes[-1]
        return values

    # -- classification ----------------------------------------------------
    def detect(
        self, series: MarketSeries, features: Optional[FeatureVector] = None
    ) -> RegimeResult:
        """Classify the regime at the end of ``series``.

        ``series`` must already be truncated to the decision timestamp.
        """
        s = self.settings
        ts = series[-1].close_time if len(series) else 0
        values = self._compute(series, features)
        candles = series.candles

        if len(candles) < 30:
            return RegimeResult(
                Regime.UNCERTAIN, 0.0,
                {"reason": 0.0, "bars": float(len(candles))}, ts,
            )

        adx_value = _safe(values.get("momentum.adx_14"))
        rsi = _safe(values.get("momentum.rsi_14"))
        atr_value = _safe(values.get("volatility.atr_14"))
        close = _safe(values.get("price.close"), candles[-1].close)
        ema21 = _safe(values.get("trend.ema_21"))
        ema50 = _safe(values.get("trend.ema_50"))
        slope = _safe(values.get("trend.ema_slope"), 0.0)

        # --- volatility percentile ---------------------------------------
        atr_series: List[float] = []
        try:
            from ..features.indicators import atr as atr_indicator
            atr_series = [v for v in atr_indicator(series.highs, series.lows, series.closes, 14)[-250:] if v == v]
        except Exception:
            atr_series = []
        atr_percentile = 0.5
        if atr_series and math.isfinite(atr_value):
            below = sum(1 for v in atr_series if v <= atr_value)
            atr_percentile = clamp(safe_div(below, len(atr_series), 0.5))

        # --- range compression / expansion --------------------------------
        recent = candles[-10:]
        longer = candles[-40:] if len(candles) >= 40 else candles
        recent_span = max(c.high for c in recent) - min(c.low for c in recent)
        longer_span = max(c.high for c in longer) - min(c.low for c in longer)
        compression = safe_div(recent_span, longer_span, 1.0)

        # --- directional bias ---------------------------------------------
        bias = 0.0
        if math.isfinite(ema21) and math.isfinite(ema50) and ema50 != 0:
            bias += 1.0 if ema21 > ema50 else -1.0
        if math.isfinite(close) and math.isfinite(ema50) and ema50 != 0:
            bias += 1.0 if close > ema50 else -1.0
        if math.isfinite(rsi):
            if rsi > 55:
                bias += 0.5
            elif rsi < 45:
                bias -= 0.5
        if slope > 0.05:
            bias += 0.5
        elif slope < -0.05:
            bias -= 0.5

        trend_score = _safe(values.get("structure.trend_score"), 0.0)
        bias += 0.5 * trend_score

        supporting: Dict[str, float] = {
            "adx_14": adx_value,
            "rsi_14": rsi,
            "atr_14": atr_value,
            "atr_percentile": atr_percentile,
            "ema_bias": bias,
            "ema_slope": slope,
            "range_compression": compression,
            "trend_score": trend_score,
        }

        # --- decision tree -------------------------------------------------
        # Volatility extremes are checked first: a volatility spike makes any
        # trend label unreliable because stops and targets are meaningless.
        if atr_percentile >= 0.95:
            regime = Regime.HIGH_VOLATILITY
            confidence = clamp(0.45 + (atr_percentile - 0.95) * 4.0)
        elif math.isfinite(adx_value) and adx_value >= s.adx_strong_threshold and abs(bias) >= 1.5:
            regime = Regime.STRONG_UPTREND if bias > 0 else Regime.STRONG_DOWNTREND
            confidence = clamp(0.5 + (adx_value - s.adx_strong_threshold) / 40.0 + abs(bias) / 10.0)
        elif math.isfinite(adx_value) and adx_value >= s.adx_trend_threshold and abs(bias) >= 0.5:
            regime = Regime.WEAK_UPTREND if bias > 0 else Regime.WEAK_DOWNTREND
            confidence = clamp(0.35 + (adx_value - s.adx_trend_threshold) / 40.0 + abs(bias) / 12.0)
        elif compression >= 1.35 and atr_percentile >= 0.6:
            # A sudden expansion out of a compressed range.
            regime = Regime.BREAKOUT
            confidence = clamp(0.30 + (compression - 1.35))
        elif compression <= s.range_compression_ratio and atr_percentile <= s.atr_low_percentile:
            regime = Regime.LOW_VOLATILITY
            confidence = clamp(0.35 + (s.range_compression_ratio - compression))
        elif math.isfinite(adx_value) and adx_value < s.adx_trend_threshold and abs(bias) < 1.0:
            regime = Regime.RANGE
            # Confidence in RANGE grows as ADX and bias both shrink.
            confidence = clamp(0.35 + (s.adx_trend_threshold - adx_value) / 40.0 + (1.0 - min(abs(bias), 1.0)) * 0.2)
        else:
            regime = Regime.UNCERTAIN
            confidence = clamp(0.25)

        # A weak label that only just cleared its threshold should not be
        # reported as decided - that is precisely the case where NO-TRADE is
        # the better answer.
        if confidence < s.min_confidence:
            supporting["downgraded_from"] = float(list(Regime).index(regime))
            regime = Regime.UNCERTAIN

        return RegimeResult(regime, clamp(confidence), supporting, ts)
