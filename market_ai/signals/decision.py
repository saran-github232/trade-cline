"""The decision gate: the final UP / DOWN / NO-TRADE authority.

Philosophy
----------
NO-TRADE is the **default**.  A directional call must be *earned* by passing
every check below.  Each check that fails is recorded as a named veto reason,
so the UI can always answer "why did you not trade?" - hiding uncertainty is
considered a defect, not a feature.

The gate is intentionally the only place that emits a :class:`Decision`.  The
ensemble produces probabilities; the gate converts them into an action under
risk and reliability constraints.  Nothing downstream may upgrade a NO-TRADE
into a directional call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..config import Config, DecisionConfig
from ..types import DataQuality, Decision, Direction, ProbabilitySet, Regime, Signal
from ..utils.logging import get_logger
from ..utils.stats import clamp, safe_div

__all__ = ["GateContext", "GateResult", "DecisionGate"]

_log = get_logger(__name__)


@dataclass
class GateContext:
    """Everything the gate needs to make an auditable decision."""

    probabilities: ProbabilitySet
    confidence: float
    agreement: float
    timestamp: int
    regime: Regime = Regime.UNCERTAIN
    regime_confidence: float = 0.0
    quality: DataQuality = DataQuality.OK
    feature_coverage: float = 1.0
    model_available: bool = True
    model_degraded: bool = False
    historical_support: int = 0
    volatility_percentile: float = 0.5
    out_of_distribution: bool = False
    signals: Sequence[Signal] = field(default_factory=list)
    notes: Sequence[str] = field(default_factory=list)


@dataclass
class GateResult:
    """The decision plus the full, explicit reasoning trail."""

    decision: Decision
    confidence: float
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    probability_margin: float = 0.0
    gate_version: str = "gate-1.0.0"

    @property
    def is_trade(self) -> bool:
        return self.decision is not Decision.NO_TRADE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": round(self.confidence, 6),
            "reasons": list(self.reasons),
            "vetoes": list(self.vetoes),
            "probability_margin": round(self.probability_margin, 6),
            "gate_version": self.gate_version,
        }


class DecisionGate:
    """Converts ensemble probabilities into UP / DOWN / NO-TRADE."""

    version = "gate-1.0.0"

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.settings: DecisionConfig = self.config.decision

    # -- individual checks -------------------------------------------------
    def _vetoes(self, ctx: GateContext) -> List[str]:
        """Every hard reason to abstain.  Order is deliberate: the most
        fundamental problems are reported first so the UI reads sensibly."""
        vetoes: List[str] = []
        s = self.settings

        if ctx.quality is DataQuality.BAD:
            vetoes.append("DATA_BAD: candle data failed validation")
        elif ctx.quality is DataQuality.DEGRADED and s.min_feature_coverage > 0.9:
            vetoes.append("DATA_DEGRADED: data quality below the validated band")

        if not ctx.model_available:
            vetoes.append("MODEL_UNAVAILABLE: no champion model could produce a probability")
        if ctx.model_degraded:
            vetoes.append("MODEL_DEGRADED: champion flagged as degraded")

        if ctx.regime is Regime.UNCERTAIN:
            vetoes.append("REGIME_UNKNOWN: market regime is UNCERTAIN")
        elif ctx.regime_confidence < self.config.regime.min_confidence:
            vetoes.append(
                f"REGIME_LOW_CONFIDENCE: regime confidence {ctx.regime_confidence:.2f} "
                f"below {self.config.regime.min_confidence:.2f}"
            )

        if ctx.feature_coverage < s.min_feature_coverage:
            vetoes.append(
                f"FEATURES_INCOMPLETE: coverage {ctx.feature_coverage:.2f} "
                f"below {s.min_feature_coverage:.2f}"
            )

        if ctx.agreement < s.min_agreement:
            vetoes.append(
                f"SIGNALS_CONFLICT: agreement {ctx.agreement:.2f} below {s.min_agreement:.2f}"
            )

        if ctx.confidence < s.min_confidence:
            vetoes.append(
                f"CONFIDENCE_INSUFFICIENT: {ctx.confidence:.2f} below {s.min_confidence:.2f}"
            )

        if ctx.out_of_distribution:
            vetoes.append("OUT_OF_DISTRIBUTION: inputs outside the validated feature range")

        if ctx.historical_support < s.min_historical_support:
            vetoes.append(
                f"HISTORICAL_SUPPORT_INSUFFICIENT: only {ctx.historical_support} "
                f"resolved samples, need {s.min_historical_support}"
            )

        vol = ctx.volatility_percentile
        if math.isfinite(vol) and (
            vol < s.volatility_percentile_floor or vol > s.volatility_percentile_ceiling
        ):
            vetoes.append(
                f"VOLATILITY_OUT_OF_RANGE: percentile {vol:.3f} outside validated band "
                f"[{s.volatility_percentile_floor:.3f}, {s.volatility_percentile_ceiling:.3f}]"
            )
        return vetoes

    # -- main ---------------------------------------------------------------
    def decide(self, ctx: GateContext) -> GateResult:
        """Return the decision plus an explicit audit trail."""
        probs = ctx.probabilities
        reasons: List[str] = []
        vetoes = self._vetoes(ctx)

        leading = max(probs.p_up, probs.p_down)
        margin = leading - probs.p_no_trade

        if vetoes:
            return GateResult(
                decision=Decision.NO_TRADE,
                confidence=ctx.confidence,
                reasons=["abstaining: " + v.split(":")[0].lower().replace("_", " ") for v in vetoes],
                vetoes=vetoes,
                probability_margin=margin,
                gate_version=self.version,
            )

        if probs.p_no_trade >= leading:
            reasons.append(
                f"NO-TRADE probability {probs.p_no_trade:.2f} dominates "
                f"(up={probs.p_up:.2f}, down={probs.p_down:.2f})"
            )
            return GateResult(
                decision=Decision.NO_TRADE,
                confidence=ctx.confidence,
                reasons=reasons,
                vetoes=["NO_TRADE_DOMINANT: abstention probability is the largest"],
                probability_margin=margin,
                gate_version=self.version,
            )

        if margin < self.settings.min_probability_margin:
            reasons.append(
                f"margin {margin:.3f} below required {self.settings.min_probability_margin:.3f}"
            )
            return GateResult(
                decision=Decision.NO_TRADE,
                confidence=ctx.confidence,
                reasons=reasons,
                vetoes=["MARGIN_TOO_SMALL: directional edge is within noise"],
                probability_margin=margin,
                gate_version=self.version,
            )

        decision = Decision.UP if probs.p_up > probs.p_down else Decision.DOWN
        reasons.append(
            f"{decision.value}: p={leading:.3f}, margin={margin:.3f}, "
            f"agreement={ctx.agreement:.2f}, confidence={ctx.confidence:.2f}"
        )
        reasons.append(f"regime={ctx.regime.value} (conf {ctx.regime_confidence:.2f})")
        reasons.append(f"data quality={ctx.quality.value}, feature coverage={ctx.feature_coverage:.2f}")
        return GateResult(
            decision=decision,
            confidence=ctx.confidence,
            reasons=reasons,
            vetoes=[],
            probability_margin=margin,
            gate_version=self.version,
        )

    def explain(self, result: GateResult) -> str:
        """Human-readable one-liner for logs and the UI."""
        if result.is_trade:
            return f"{result.decision.value} - " + "; ".join(result.reasons)
        joined = "; ".join(result.vetoes) if result.vetoes else "insufficient evidence"
        return f"NO-TRADE - {joined}"
