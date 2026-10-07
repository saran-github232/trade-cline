"""Concrete evidence providers.

Each provider is a small, explainable rule over already-computed features.
They are intentionally *not* learned models: their job is to supply the
ensemble with independent, human-auditable opinions so that a disagreement
between them is meaningful.  The learned component lives in the ML signal.

All providers read features through dotted names, so they depend only on
``FeatureVector`` and never on the feature engine's internals.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

from ..types import Direction, Signal
from ..utils.stats import clamp, safe_div
from .base import (
    SignalContext,
    make_signal,
    unavailable_signal,
)

__all__ = [
    "TechnicalSignalProvider",
    "PriceActionSignalProvider",
    "StructureSignalProvider",
    "RegimeSignalProvider",
    "MLSignalProvider",
    "VisionSignalProvider",
    "MultiTimeframeSignalProvider",
    "default_providers",
]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return fval if math.isfinite(fval) else default


def _squash(score: float, scale: float = 1.0) -> float:
    """Map an unbounded score onto (0, 1) with a logistic curve.

    Keeping every provider on the same probability scale is what makes them
    combinable.  ``scale`` sets how many score units correspond to roughly a
    doubling of the odds.
    """
    if scale <= 0:
        scale = 1.0
    try:
        return clamp(1.0 / (1.0 + math.exp(-score / scale)))
    except OverflowError:
        return 0.0 if score < 0 else 1.0


class TechnicalSignalProvider:
    """Trend + momentum + mean-reversion evidence from classic indicators."""

    name = "technical"

    def compute(self, context: SignalContext) -> Signal:
        ts = context.as_of
        f = context.feature
        score = 0.0
        evidence: Dict[str, Any] = {}
        contributors = 0

        # EMA stack: price above a rising stack is the cleanest trend evidence.
        ema9 = _finite(f("trend.ema_9"), math.nan)
        ema21 = _finite(f("trend.ema_21"), math.nan)
        ema50 = _finite(f("trend.ema_50"), math.nan)
        close = _finite(f("price.close"), math.nan)
        if all(math.isfinite(v) for v in (ema9, ema21, ema50, close)) and ema50 != 0:
            stack = (close - ema50) / abs(ema50)
            spread = (ema9 - ema21) / abs(ema21) if ema21 else 0.0
            score += 4.0 * stack + 3.0 * spread
            contributors += 1
            evidence["ema_stack"] = round(stack, 6)
            evidence["ema_spread"] = round(spread, 6)

        rsi = _finite(f("momentum.rsi_14"), math.nan)
        if math.isfinite(rsi):
            # Centred at 50 with a mild contrarian tilt beyond the extremes:
            # overbought in a trend is continuation, overbought in a range is
            # exhaustion, and the regime provider resolves that ambiguity.
            score += (rsi - 50.0) / 15.0
            contributors += 1
            evidence["rsi_14"] = round(rsi, 3)

        hist = _finite(f("momentum.macd_hist"), math.nan)
        if math.isfinite(hist):
            atr = _finite(f("volatility.atr_14"), math.nan)
            if math.isfinite(atr) and atr > 0:
                # Normalising by ATR keeps the MACD contribution comparable
                # across assets with very different absolute price scales.
                score += clamp(hist / atr, -1.5, 1.5) * 1.2
                contributors += 1
                evidence["macd_hist_atr_norm"] = round(hist / atr, 4)

        k = _finite(f("momentum.stoch_k"), math.nan)
        if math.isfinite(k):
            score += (k - 50.0) / 30.0
            contributors += 1
            evidence["stoch_k"] = round(k, 3)

        pct_b = _finite(f("volatility.bb_pct_b"), math.nan)
        if math.isfinite(pct_b):
            # Only mildly mean-reverting: strong bands in a trend must not be
            # treated as a reversal signal.
            score += (0.5 - pct_b) * 0.6
            contributors += 1
            evidence["bb_pct_b"] = round(pct_b, 4)

        if contributors == 0:
            return unavailable_signal(self.name, ts, "no technical features available")

        # scale=3.5 keeps a strong-but-not-extreme reading near 0.75-0.80.
        # Deliberately conservative: an over-confident provider is worse than a
        # useless one, because it drags the ensemble past the NO-TRADE gate.
        probability = _squash(score, scale=3.5)
        # Confidence grows with the number of independent contributors, and is
        # damped by poor data quality.
        confidence = clamp(0.35 + 0.13 * contributors, 0.0, 0.9) * clamp(context.quality_score, 0.0, 1.0)
        evidence["contributors"] = contributors
        evidence["raw_score"] = round(score, 4)
        return make_signal(self.name, probability, confidence, ts, evidence=evidence)


class PriceActionSignalProvider:
    """Candle-geometry and pattern evidence."""

    name = "price_action"

    def compute(self, context: SignalContext) -> Signal:
        ts = context.as_of
        f = context.feature
        score = 0.0
        contributors = 0
        evidence: Dict[str, Any] = {}

        body_ratio = _finite(f("pa.body_ratio"), math.nan)
        if math.isfinite(body_ratio):
            score += body_ratio * 1.2
            contributors += 1
            evidence["body_ratio"] = round(body_ratio, 4)

        wick_ratio = _finite(f("pa.body_wick_ratio"), math.nan)
        if math.isfinite(wick_ratio):
            # A long lower wick (negative skew) is supportive, and vice versa.
            score += clamp(wick_ratio, -3.0, 3.0) * 0.7
            contributors += 1
            evidence["body_wick_ratio"] = round(wick_ratio, 4)

        if _finite(f("pa.bullish_engulfing"), 0.0) > 0.5:
            score += 1.4
            contributors += 1
            evidence["bullish_engulfing"] = True
        if _finite(f("pa.bearish_engulfing"), 0.0) > 0.5:
            score -= 1.4
            contributors += 1
            evidence["bearish_engulfing"] = True
        if _finite(f("pa.bullish_rejection"), 0.0) > 0.5:
            score += 0.9
            contributors += 1
            evidence["bullish_rejection"] = True
        if _finite(f("pa.bearish_rejection"), 0.0) > 0.5:
            score -= 0.9
            contributors += 1
            evidence["bearish_rejection"] = True
        if _finite(f("pa.inside_bar"), 0.0) > 0.5:
            # Compression is genuinely directionless: add no score, but note it.
            evidence["inside_bar"] = True
        if _finite(f("pa.range_expansion"), 0.0) > 0.5:
            direction_hint = _finite(f("pa.last_candle_direction"), 0.0)
            score += direction_hint * 0.6
            contributors += 1
            evidence["range_expansion"] = True

        bull_seq = _finite(f("pa.bullish_sequence"), 0.0)
        bear_seq = _finite(f("pa.bearish_sequence"), 0.0)
        if bull_seq or bear_seq:
            score += 0.5 * (bull_seq - bear_seq)
            contributors += 1
            evidence["bullish_sequence"] = bull_seq
            evidence["bearish_sequence"] = bear_seq

        if contributors == 0:
            return unavailable_signal(self.name, ts, "no price-action features available")

        probability = _squash(score, scale=2.4)
        confidence = clamp(0.3 + 0.12 * contributors, 0.0, 0.85) * clamp(context.quality_score, 0.0, 1.0)
        evidence["contributors"] = contributors
        return make_signal(self.name, probability, confidence, ts, evidence=evidence)


class StructureSignalProvider:
    """Market-structure evidence: swings, breaks, support/resistance."""

    name = "structure"

    def compute(self, context: SignalContext) -> Signal:
        ts = context.as_of
        f = context.feature
        score = 0.0
        contributors = 0
        evidence: Dict[str, Any] = {}

        hh = _finite(f("structure.higher_high"), 0.0)
        hl = _finite(f("structure.higher_low"), 0.0)
        lh = _finite(f("structure.lower_high"), 0.0)
        ll = _finite(f("structure.lower_low"), 0.0)
        if any((hh, hl, lh, ll)):
            score += 1.1 * (hh + hl) - 1.1 * (lh + ll)
            contributors += 1
            evidence.update({"higher_high": hh, "higher_low": hl, "lower_high": lh, "lower_low": ll})

        breakout = _finite(f("structure.breakout_up"), 0.0)
        breakdown = _finite(f("structure.breakout_down"), 0.0)
        if breakout or breakdown:
            score += 1.3 * breakout - 1.3 * breakdown
            contributors += 1
            evidence.update({"breakout_up": breakout, "breakout_down": breakdown})

        failed_up = _finite(f("structure.failed_breakout_up"), 0.0)
        failed_down = _finite(f("structure.failed_breakout_down"), 0.0)
        if failed_up or failed_down:
            # A failed breakout is evidence for the OPPOSITE direction.
            score += -1.2 * failed_up + 1.2 * failed_down
            contributors += 1
            evidence.update({"failed_breakout_up": failed_up, "failed_breakout_down": failed_down})

        dist_res = _finite(f("structure.distance_to_resistance_atr"), math.nan)
        dist_sup = _finite(f("structure.distance_to_support_atr"), math.nan)
        if math.isfinite(dist_res) and math.isfinite(dist_sup):
            # Being far from resistance and near support is structurally bullish.
            score += clamp((dist_res - dist_sup) / 2.0, -2.0, 2.0) * 0.6
            contributors += 1
            evidence.update({
                "distance_to_resistance_atr": round(dist_res, 3),
                "distance_to_support_atr": round(dist_sup, 3),
            })

        if contributors == 0:
            return unavailable_signal(self.name, ts, "no structure features available")

        probability = _squash(score, scale=2.6)
        confidence = clamp(0.32 + 0.13 * contributors, 0.0, 0.88) * clamp(context.quality_score, 0.0, 1.0)
        evidence["contributors"] = contributors
        return make_signal(self.name, probability, confidence, ts, evidence=evidence)


class RegimeSignalProvider:
    """Directional implication of the detected regime.

    The regime provider deliberately reports LOW confidence in RANGE and
    UNCERTAIN regimes.  That is how a ranging market pushes the ensemble
    towards NO-TRADE instead of manufacturing a direction.
    """

    name = "regime"

    #: Prior P(UP) and confidence for each regime label.
    _PRIORS: Dict[str, tuple] = {
        "STRONG_UPTREND": (0.68, 0.80),
        "WEAK_UPTREND": (0.58, 0.55),
        "STRONG_DOWNTREND": (0.32, 0.80),
        "WEAK_DOWNTREND": (0.42, 0.55),
        "BREAKOUT": (0.55, 0.35),
        "RANGE": (0.50, 0.15),
        "HIGH_VOLATILITY": (0.50, 0.12),
        "LOW_VOLATILITY": (0.50, 0.18),
        "UNCERTAIN": (0.50, 0.05),
    }

    def compute(self, context: SignalContext) -> Signal:
        ts = context.as_of
        regime_result = context.regime
        if regime_result is None:
            return unavailable_signal(self.name, ts, "regime unavailable")
        label = getattr(regime_result, "regime", None)
        label = getattr(label, "value", str(label))
        prior_p, prior_c = self._PRIORS.get(label, (0.5, 0.05))
        regime_conf = clamp(getattr(regime_result, "confidence", 0.0))

        # Blend the regime prior with any directional evidence the detector
        # itself surfaced, so a "weak uptrend with strong momentum" is treated
        # differently from a "weak uptrend with fading momentum".
        features = dict(getattr(regime_result, "supporting_features", {}) or {})
        momentum = _finite(features.get("momentum.rsi_14"), math.nan)
        probability = prior_p
        if math.isfinite(momentum):
            tilt = clamp((momentum - 50.0) / 100.0, -0.1, 0.1)
            probability = clamp(prior_p + tilt)

        confidence = clamp(prior_c * (0.4 + 0.6 * regime_conf))
        return make_signal(
            self.name, probability, confidence, ts,
            evidence={"regime": label, "regime_confidence": regime_conf, "supporting": features},
        )


class MLSignalProvider:
    """The supervised model's calibrated probability."""

    name = "ml"

    def compute(self, context: SignalContext) -> Signal:
        ts = context.as_of
        probability = context.ml_probability
        if probability is None:
            return unavailable_signal(self.name, ts, "no model available")
        try:
            p = float(probability)
        except (TypeError, ValueError):
            return unavailable_signal(self.name, ts, "model output not numeric")
        if not math.isfinite(p):
            return unavailable_signal(self.name, ts, "model output not finite")

        # Confidence scales with how far the model is from a coin flip.  A
        # model that outputs 0.501 is not evidence, and must not be counted.
        edge = abs(p - 0.5) * 2.0
        coverage = clamp(context.quality_score)
        confidence = clamp(edge * 1.35) * coverage
        return make_signal(
            self.name, p, confidence, ts,
            evidence={"model_probability": round(p, 6), "edge": round(edge, 4)},
        )


class VisionSignalProvider:
    """Chart-vision evidence, hard-capped in influence.

    Vision is the least reliable provider (it is a perceptual model over a
    rendered image), so its confidence is capped by ``VisionConfig`` and it is
    never allowed to dominate.  When the vision result is degraded, its
    confidence is halved rather than dropped, so it still contributes but
    cannot swing the gate.
    """

    name = "vision"

    def __init__(self, max_influence: float = 0.10) -> None:
        self.max_influence = clamp(max_influence, 0.0, 1.0)

    def compute(self, context: SignalContext) -> Signal:
        ts = context.as_of
        vision = context.vision
        if vision is None:
            return unavailable_signal(self.name, ts, "vision unavailable")
        bias = getattr(vision, "direction_bias", None)
        bias_value = getattr(bias, "value", str(bias))
        confidence = clamp(getattr(vision, "confidence", 0.0))
        if getattr(vision, "degraded", False):
            confidence *= 0.5
        breakout = clamp(_finite(getattr(vision, "breakout_probability", 0.5), 0.5))
        reversal = clamp(_finite(getattr(vision, "reversal_probability", 0.5), 0.5))

        if bias_value == "UP":
            probability = 0.5 + 0.25 * confidence + 0.10 * (breakout - 0.5)
        elif bias_value == "DOWN":
            probability = 0.5 - 0.25 * confidence - 0.10 * (reversal - 0.5)
        else:
            probability = 0.5
            confidence *= 0.3

        return make_signal(
            self.name, clamp(probability), clamp(confidence * self.max_influence * 4.0), ts,
            evidence={
                "bias": bias_value,
                "trend": getattr(vision, "trend", ""),
                "structure": getattr(vision, "structure", ""),
                "breakout_probability": breakout,
                "reversal_probability": reversal,
                "degraded": bool(getattr(vision, "degraded", False)),
                "provider": getattr(vision, "provider", "unknown"),
            },
        )


class MultiTimeframeSignalProvider:
    """Hierarchical agreement across timeframes.

    The higher timeframe sets the bias; the lower timeframe can only add or
    remove conviction, never invert the higher-timeframe regime.
    """

    name = "multi_timeframe"

    def compute(self, context: SignalContext) -> Signal:
        ts = context.as_of
        mtf = context.mtf
        if mtf is None:
            return unavailable_signal(self.name, ts, "multi-timeframe unavailable")
        aligned = getattr(mtf, "aligned_direction", None)
        aligned_value = getattr(aligned, "value", str(aligned))
        agreement = clamp(getattr(mtf, "agreement", 0.0))
        conflict = clamp(getattr(mtf, "conflict", 0.0))

        if aligned_value == "UP":
            probability = 0.5 + 0.35 * agreement
        elif aligned_value == "DOWN":
            probability = 0.5 - 0.35 * agreement
        else:
            probability = 0.5

        # Confidence is driven by agreement and penalised by conflict.  When
        # the timeframes disagree outright, confidence collapses to near zero.
        confidence = clamp(agreement * (1.0 - conflict))
        return make_signal(
            self.name, clamp(probability), confidence, ts,
            evidence={
                "aligned_direction": aligned_value,
                "agreement": agreement,
                "conflict": conflict,
                "higher": _view_summary(getattr(mtf, "higher", None)),
                "middle": _view_summary(getattr(mtf, "middle", None)),
                "lower": _view_summary(getattr(mtf, "lower", None)),
            },
        )


def _view_summary(view: Any) -> Optional[Dict[str, Any]]:
    if view is None:
        return None
    direction = getattr(view, "direction", None)
    return {
        "timeframe": getattr(getattr(view, "timeframe", None), "value", None),
        "direction": getattr(direction, "value", str(direction)),
        "strength": _finite(getattr(view, "strength", 0.0)),
    }


def default_providers(max_vision_influence: float = 0.10) -> List[Any]:
    """The standard provider set, in a stable order."""
    return [
        TechnicalSignalProvider(),
        PriceActionSignalProvider(),
        StructureSignalProvider(),
        RegimeSignalProvider(),
        MLSignalProvider(),
        VisionSignalProvider(max_influence=max_vision_influence),
        MultiTimeframeSignalProvider(),
    ]
