"""Multi-signal ensemble.

Design rationale
----------------
The ensemble does **not** simply average provider probabilities.  Averaging
is wrong here for two reasons:

1. Providers are not equally trustworthy, and a provider's reliability varies
   with conditions.  A weighted combination that also multiplies by each
   provider's own confidence handles that.
2. When providers disagree, the correct output is *less directional mass*,
   not a probability near 0.5.  A 0.5 output is indistinguishable from "no
   opinion", and the decision gate could still read it as a weak UP or DOWN.

So the model is deliberately *mass-based*::

    p_up       = directional_mass * mean_p
    p_down     = directional_mass * (1 - mean_p)
    p_no_trade = 1 - directional_mass

``directional_mass`` is what the providers collectively earn, shrunk by
agreement, regime certainty, data quality and the reasoning supervisor.  When
evidence is weak or contradictory, mass collapses towards zero and NO-TRADE
dominates automatically - it is the structural default, not a special case.

Weights are initialisation values only.  ``market_ai.ensemble.weights``
measures whether different weights generalise better on unseen data, and the
promoted vector is stored alongside the model version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config, EnsembleConfig
from ..signals.base import SignalContext
from ..types import Decision, Direction, ProbabilitySet, Signal, Timeframe
from ..utils.logging import get_logger
from ..utils.stats import clamp, mean, safe_div, stdev

__all__ = ["EnsembleResult", "EnsembleEngine", "disagreement_score", "agreement_score"]

_log = get_logger(__name__)

#: Providers that carry weight mass.  Everything else acts as a modifier.
WEIGHTED_PROVIDERS: Tuple[str, ...] = (
    "technical",
    "price_action",
    "structure",
    "regime",
    "ml",
    "vision",
)

#: Modifiers can only *reduce* directional mass, never increase it.
MODIFIER_PROVIDERS: Tuple[str, ...] = ("multi_timeframe",)


def disagreement_score(signals: Sequence[Signal]) -> float:
    """Weighted dispersion of provider probabilities, normalised to [0, 1].

    0.0 means every provider agrees; 1.0 means they are maximally split
    (e.g. half at 1.0 and half at 0.0).  Confidence-weighted so a provider
    that admits it has no idea does not manufacture disagreement.
    """
    usable = [(s.probability, s.confidence) for s in signals if s.available and s.confidence > 0]
    if len(usable) < 2:
        return 0.0
    total_w = sum(w for _, w in usable)
    if total_w <= 0:
        return 0.0
    centre = sum(p * w for p, w in usable) / total_w
    var = sum(w * (p - centre) ** 2 for p, w in usable) / total_w
    # A std-dev of 0.5 is the theoretical maximum for probabilities in [0,1].
    return clamp(math.sqrt(var) / 0.5)


def agreement_score(signals: Sequence[Signal]) -> float:
    """Complement of disagreement, plus a bonus for aligned direction labels."""
    disp = disagreement_score(signals)
    directional = [s for s in signals if s.available and s.direction is not Direction.NEUTRAL]
    if not directional:
        return 0.0
    ups = sum(1 for s in directional if s.direction is Direction.UP)
    downs = len(directional) - ups
    label_agreement = max(ups, downs) / len(directional)
    return clamp(0.6 * (1.0 - disp) + 0.4 * label_agreement)


@dataclass
class EnsembleResult:
    """Combined view of all evidence at one decision point."""

    timestamp: int
    probabilities: ProbabilitySet
    signals: List[Signal]
    agreement: float
    disagreement: float
    directional_mass: float
    mean_probability: float
    weights_used: Dict[str, float]
    modifiers: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    version: str = "ens-1.0.0"

    @property
    def decision(self) -> Decision:
        return self.probabilities.best()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "probabilities": self.probabilities.as_dict(),
            "agreement": round(self.agreement, 6),
            "disagreement": round(self.disagreement, 6),
            "directional_mass": round(self.directional_mass, 6),
            "mean_probability": round(self.mean_probability, 6),
            "weights_used": {k: round(v, 6) for k, v in self.weights_used.items()},
            "modifiers": {k: round(v, 6) for k, v in self.modifiers.items()},
            "notes": list(self.notes),
            "version": self.version,
            "signals": [s.to_dict() for s in self.signals],
        }


class EnsembleEngine:
    """Combines provider signals into a single probability triple."""

    version = "ens-1.0.0"

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.config = config or Config()
        self.ensemble_config: EnsembleConfig = self.config.ensemble
        self.weights: Dict[str, float] = dict(weights or self.ensemble_config.weights)

    # -- weight handling ---------------------------------------------------
    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Install a weight vector (e.g. one promoted from validation)."""
        merged = dict(self.weights)
        for key, value in weights.items():
            try:
                merged[str(key)] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
        self.weights = merged

    def _normalised_weights(self) -> Dict[str, float]:
        total = sum(self.weights.get(name, 0.0) for name in WEIGHTED_PROVIDERS)
        if total <= 0:
            return {name: 1.0 / len(WEIGHTED_PROVIDERS) for name in WEIGHTED_PROVIDERS}
        return {name: self.weights.get(name, 0.0) / total for name in WEIGHTED_PROVIDERS}

    # -- core --------------------------------------------------------------
    def combine(
        self,
        signals: Sequence[Signal],
        *,
        timestamp: int,
        context: Optional[SignalContext] = None,
        reasoning: Any = None,
    ) -> EnsembleResult:
        """Fold provider signals into P(UP), P(DOWN), P(NO_TRADE)."""
        notes: List[str] = []
        by_source = {s.source: s for s in signals}
        base_weights = self._normalised_weights()

        # --- 1. evidence mass -------------------------------------------------
        # Each provider contributes weight * its own confidence.  A provider
        # that is unsure simply contributes less mass, which is exactly the
        # behaviour we want.
        total_mass = 0.0
        weighted_p = 0.0
        weights_used: Dict[str, float] = {}
        for name in WEIGHTED_PROVIDERS:
            signal = by_source.get(name)
            if signal is None or not signal.available or signal.confidence <= 0:
                weights_used[name] = 0.0
                continue
            contribution = base_weights.get(name, 0.0) * signal.confidence
            total_mass += contribution
            weighted_p += contribution * signal.probability
            weights_used[name] = contribution

        available = [s for s in signals if s.available and s.confidence > 0]
        missing = [name for name in WEIGHTED_PROVIDERS if weights_used.get(name, 0.0) <= 0]
        if missing:
            notes.append(f"providers unavailable: {', '.join(missing)}")

        if total_mass <= 0:
            # No evidence at all.  This is the canonical NO-TRADE case and it
            # must be unambiguous, not a coin flip.
            notes.append("no usable evidence from any weighted provider")
            return EnsembleResult(
                timestamp=timestamp,
                probabilities=ProbabilitySet(0.0, 0.0, 1.0),
                signals=list(signals),
                agreement=0.0,
                disagreement=0.0,
                directional_mass=0.0,
                mean_probability=0.5,
                weights_used=weights_used,
                notes=notes,
                version=self.version,
            )

        mean_p = weighted_p / total_mass
        directional_mass = clamp(total_mass)
        disagreement = disagreement_score(available)
        agreement = agreement_score(available)

        # --- 2. shrink mass for disagreement ---------------------------------
        # Disagreement removes directional mass; it never flips the direction.
        # A floor keeps a genuinely unanimous-but-faint signal alive.
        agreement_factor = clamp(agreement, 0.0, 1.0)
        directional_mass *= agreement_factor
        if disagreement > self.ensemble_config.max_disagreement:
            notes.append(f"high disagreement ({disagreement:.2f})")

        # --- 3. regime certainty ---------------------------------------------
        regime_factor = 1.0
        if context is not None and context.regime is not None:
            regime_label = getattr(getattr(context.regime, "regime", None), "value", "UNCERTAIN")
            regime_conf = clamp(getattr(context.regime, "confidence", 0.0))
            if regime_label == "UNCERTAIN":
                regime_factor = 0.35
                notes.append("regime is UNCERTAIN -> suppressing directional mass")
            else:
                regime_factor = 0.55 + 0.45 * regime_conf

        # --- 4. data quality --------------------------------------------------
        quality_factor = 1.0
        if context is not None:
            quality_factor = clamp(context.quality_score, 0.0, 1.0)
            if quality_factor < 0.5:
                notes.append(f"degraded data quality ({quality_factor:.2f})")

        # --- 5. multi-timeframe modifier -------------------------------------
        mtf_factor = 1.0
        mtf_signal = by_source.get("multi_timeframe")
        modifiers: Dict[str, float] = {}
        if mtf_signal is not None and mtf_signal.available:
            conflict = clamp((mtf_signal.evidence or {}).get("conflict", 0.0))
            agreement_mtf = clamp((mtf_signal.evidence or {}).get("agreement", 0.0))
            strength = clamp(self.ensemble_config.mtf_modifier_strength, 0.0, 1.0)
            # Conflict shrinks mass; alignment restores a little of it.
            mtf_factor = clamp(1.0 - strength * conflict + 0.5 * strength * agreement_mtf * (1.0 - conflict))
            modifiers["multi_timeframe"] = mtf_factor
            if conflict > 0.5:
                notes.append(f"timeframe conflict ({conflict:.2f})")

        # --- 6. reasoning modifier --------------------------------------------
        reasoning_factor = 1.0
        if reasoning is not None:
            contradictions = list(getattr(reasoning, "contradictions", []) or [])
            adjustment = float(getattr(reasoning, "confidence_adjustment", 0.0) or 0.0)
            # The supervisor can only ever reduce mass (adjustment <= 0).
            adjustment = min(0.0, adjustment)
            strength = clamp(self.ensemble_config.reasoning_modifier_strength, 0.0, 1.0)
            penalty = strength * len(contradictions) * 0.5
            reasoning_factor = clamp(1.0 + adjustment - penalty, 0.0, 1.0)
            modifiers["reasoning"] = reasoning_factor
            if contradictions:
                notes.append(f"supervisor flagged {len(contradictions)} contradiction(s)")

        directional_mass = clamp(
            directional_mass * regime_factor * quality_factor * mtf_factor * reasoning_factor
        )

        # --- 7. assemble -------------------------------------------------------
        p_up = directional_mass * mean_p
        p_down = directional_mass * (1.0 - mean_p)
        p_no_trade = max(0.0, 1.0 - directional_mass)

        if directional_mass < 0.15:
            notes.append("directional mass below actionable level")

        return EnsembleResult(
            timestamp=timestamp,
            probabilities=ProbabilitySet(p_up, p_down, p_no_trade),
            signals=list(signals),
            agreement=agreement,
            disagreement=disagreement,
            directional_mass=directional_mass,
            mean_probability=mean_p,
            weights_used=weights_used,
            modifiers=modifiers,
            notes=notes,
            version=self.version,
        )

    def combine_context(self, context: SignalContext) -> EnsembleResult:
        """Convenience wrapper for a context that already carries its signals."""
        signals = list(context.extra.get("signals", []))
        return self.combine(
            signals, timestamp=context.as_of, context=context, reasoning=context.reasoning
        )
