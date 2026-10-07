"""Signal provider contract.

A *signal* is one independent opinion about the near-term direction.  The
ensemble deliberately keeps providers decoupled: each reads only the context
it needs and returns a :class:`Signal`, so a failing provider degrades to
``available=False`` instead of taking down the pipeline.

Every provider reports two different numbers and they must not be confused:

``probability``
    P(price closes higher over the horizon).  This is the opinion.
``confidence``
    How much the provider trusts its own opinion given the data quality and
    how far its inputs are from their historical norms.  This is the weight.

A provider that is uncertain must say so by lowering ``confidence``, not by
pushing ``probability`` to 0.5 and pretending that is information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from ..types import (
    Decision,
    Direction,
    FeatureVector,
    MarketSeries,
    Signal,
    Timeframe,
)
from ..utils.stats import clamp

__all__ = ["SignalContext", "SignalProvider", "direction_from_probability", "make_signal"]


@dataclass(frozen=True)
class SignalContext:
    """Everything a provider is allowed to see at a decision point.

    ``series`` is the causal history only.  There is deliberately no ``future``
    field: a provider physically cannot access post-decision data.
    """

    asset: str
    timeframe: Timeframe
    as_of: int
    series: MarketSeries
    features: FeatureVector
    regime: Any = None                 # RegimeResult
    mtf: Any = None                    # MultiTimeframeResult
    ml_probability: Optional[float] = None
    vision: Any = None                 # VisionResult
    reasoning: Any = None              # ReasoningResult
    quality_score: float = 1.0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def feature(self, name: str, default: float = float("nan")) -> float:
        return self.features.get(name, default)

    def has(self, name: str) -> bool:
        value = self.features.get(name)
        return value is not None and value == value  # NaN check without math import


class SignalProvider(Protocol):
    """Minimal contract every evidence provider implements."""

    name: str

    def compute(self, context: SignalContext) -> Signal:
        """Return this provider's opinion at the context's timestamp."""
        ...


def direction_from_probability(probability: float, *, neutral_band: float = 0.04) -> Direction:
    """Convert P(UP) into a direction, honouring an explicit neutral band.

    The band exists so that a 0.51 is reported as NEUTRAL rather than as a
    weak UP: the ensemble needs to distinguish "no opinion" from "a tiny
    opinion", because only the former should raise NO-TRADE.
    """
    p = clamp(probability)
    if p >= 0.5 + neutral_band:
        return Direction.UP
    if p <= 0.5 - neutral_band:
        return Direction.DOWN
    return Direction.NEUTRAL


def make_signal(
    source: str,
    probability: float,
    confidence: float,
    timestamp: int,
    *,
    evidence: Optional[Mapping[str, Any]] = None,
    available: bool = True,
    notes: str = "",
    neutral_band: float = 0.04,
) -> Signal:
    """Build a :class:`Signal` with a consistent direction convention."""
    p = clamp(probability)
    return Signal(
        source=source,
        direction=direction_from_probability(p, neutral_band=neutral_band),
        probability=p,
        confidence=clamp(confidence),
        timestamp=timestamp,
        evidence=dict(evidence or {}),
        available=available,
        notes=notes,
    )


def unavailable_signal(source: str, timestamp: int, reason: str) -> Signal:
    """A neutral, zero-confidence signal used when a provider fails.

    Returning this rather than raising is what makes ``MODEL FAILURE ->
    NO-TRADE`` work: the ensemble sees a provider that is simply absent.
    """
    return Signal(
        source=source,
        direction=Direction.NEUTRAL,
        probability=0.5,
        confidence=0.0,
        timestamp=timestamp,
        evidence={},
        available=False,
        notes=reason,
    )
