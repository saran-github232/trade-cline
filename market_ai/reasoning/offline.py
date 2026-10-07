"""Deterministic, rule-based reasoning supervisor.

This is the default and always-available supervisor.  It inspects the
structured ``context`` (technical / ML / vision / regime / multi-timeframe
evidence, data quality, volatility, coverage) and produces a critique that:

* flags **contradictions** between evidence sources;
* flags **weak evidence** (unavailable or low-strength sources);
* flags **abnormal conditions** (volatility outside the historical band, low
  feature coverage, degraded providers, bad data quality);
* lists **invalidating conditions**;
* reduces confidence (``confidence_adjustment`` in ``[-1, 0]``);
* recommends ``NO_TRADE`` when a hard contradiction exists and config says so.

It never invents market data, never creates a directional call and never
changes weights or risk limits - it can only push *towards* NO-TRADE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..config import Config
from ..types import Decision, Direction
from ..utils.stats import clamp
from .schema import ReasoningResult

__all__ = ["OfflineReasoningSupervisor"]


#: Context keys that may carry a single evidence source's opinion.
_EVIDENCE_KEYS = (
    "technical",
    "ml",
    "model",
    "vision",
    "regime",
    "multi_timeframe",
    "mtf",
)

#: A source at/above this strength counts as "strong" evidence.
_STRONG_THRESHOLD = 0.6
#: A source below this strength counts as "weak" evidence.
_WEAK_THRESHOLD = 0.4
#: Feature coverage below this is flagged as abnormal.
_MIN_COVERAGE = 0.80
#: Feature coverage below this is severe enough to force NO-TRADE.
_SEVERE_COVERAGE = 0.50


@dataclass(frozen=True)
class _Opinion:
    """A single evidence source's directional opinion."""

    source: str
    direction: Direction
    strength: float
    available: bool = True


# --------------------------------------------------------------------------
# context extraction helpers
# --------------------------------------------------------------------------
def _coerce_direction(value: Any) -> Direction:
    """Map a loose directional label onto :class:`Direction` (NEUTRAL default)."""
    if isinstance(value, Direction):
        return value
    if value is None:
        return Direction.NEUTRAL
    text = str(value).strip().upper()
    if text in ("UP", "LONG", "BULL", "BULLISH", "BUY"):
        return Direction.UP
    if text in ("DOWN", "SHORT", "BEAR", "BEARISH", "SELL"):
        return Direction.DOWN
    return Direction.NEUTRAL


def _coerce_decision(value: Any) -> Optional[Decision]:
    """Map a loose decision label onto :class:`Decision` (or ``None``)."""
    if value is None:
        return None
    if isinstance(value, Decision):
        return value
    text = str(value).strip().upper().replace("_", "-").replace(" ", "-")
    if text in ("UP", "LONG", "BUY"):
        return Decision.UP
    if text in ("DOWN", "SHORT", "SELL"):
        return Decision.DOWN
    if text in ("NO-TRADE", "NOTRADE", "NONE", "NEUTRAL", "FLAT", "SKIP", "HOLD"):
        return Decision.NO_TRADE
    return None


def _as_float(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fval = float(value)
    return fval if math.isfinite(fval) else None


def _strength(mapping: Mapping[str, Any]) -> float:
    """Derive a 0..1 strength from a source mapping (confidence or probability)."""
    conf = _as_float(mapping.get("confidence"))
    if conf is not None:
        return clamp(conf, 0.0, 1.0)
    prob = _as_float(mapping.get("probability"))
    if prob is not None:
        return clamp(abs(prob - 0.5) * 2.0, 0.0, 1.0)
    return 0.5


def _opinion_from_mapping(source: str, mapping: Any) -> Optional[_Opinion]:
    """Build an :class:`_Opinion` from a source mapping, or ``None``."""
    if not isinstance(mapping, Mapping):
        return None
    raw_direction = mapping.get(
        "direction", mapping.get("bias", mapping.get("direction_bias"))
    )
    if raw_direction is None:
        return None
    return _Opinion(
        source=source,
        direction=_coerce_direction(raw_direction),
        strength=_strength(mapping),
        available=bool(mapping.get("available", True)),
    )


def _collect_opinions(context: Mapping[str, Any]) -> List[_Opinion]:
    """Collect evidence opinions from the known context keys."""
    opinions: List[_Opinion] = []
    for key in _EVIDENCE_KEYS:
        opinion = _opinion_from_mapping(key, context.get(key))
        if opinion is not None:
            opinions.append(opinion)

    signals = context.get("signals")
    if isinstance(signals, (list, tuple)):
        for index, signal in enumerate(signals):
            if not isinstance(signal, Mapping):
                continue
            source = str(signal.get("source", f"signal_{index}"))
            opinion = _opinion_from_mapping(source, signal)
            if opinion is not None:
                opinions.append(opinion)
    return opinions


def _str_list(value: Any) -> List[str]:
    """Coerce a value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return []


def _dedupe(items: Sequence[str]) -> List[str]:
    """De-duplicate while preserving order."""
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out



# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------
class OfflineReasoningSupervisor:
    """Deterministic, rule-based supervisor (the default implementation)."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Store configuration; the offline supervisor needs nothing else."""
        self.config = config or Config()
        self.name = "offline"

    def review(self, *, context: Mapping[str, Any]) -> ReasoningResult:
        """Critique the evidence in ``context`` and return a safe result."""
        ctx = dict(context or {})
        base = self._base_decision(ctx)
        opinions = _collect_opinions(ctx)

        contradictions = self._contradictions(ctx, base, opinions)
        weak = self._weak_evidence(opinions)
        abnormal = self._abnormal_conditions(ctx)
        invalidating = self._invalidating_conditions(ctx)

        adjustment = self._confidence_adjustment(contradictions, weak, abnormal)
        recommendation = self._recommendation(base, contradictions, ctx, abnormal)
        explanation = self._explanation(
            base, recommendation, contradictions, weak, abnormal, adjustment
        )

        raw = {
            "source": "offline",
            "base_decision": base.value if base else None,
            "opinions": [
                {
                    "source": o.source,
                    "direction": o.direction.value,
                    "strength": o.strength,
                    "available": o.available,
                }
                for o in opinions
            ],
            "counts": {
                "contradictions": len(contradictions),
                "weak_evidence": len(weak),
                "abnormal_conditions": len(abnormal),
            },
        }

        return ReasoningResult(
            recommendation=recommendation,
            contradictions=contradictions,
            weak_evidence=weak,
            invalidating_conditions=invalidating,
            abnormal_conditions=abnormal,
            explanation=explanation,
            confidence_adjustment=adjustment,
            provider="offline",
            degraded=False,
            raw=raw,
        )

    # -- internals --------------------------------------------------------
    @staticmethod
    def _base_decision(ctx: Mapping[str, Any]) -> Optional[Decision]:
        """Extract the base decision the supervisor is reviewing."""
        return _coerce_decision(ctx.get("decision", ctx.get("base_decision")))

    def _contradictions(
        self,
        ctx: Mapping[str, Any],
        base: Optional[Decision],
        opinions: Sequence[_Opinion],
    ) -> List[str]:
        """Detect conflicting directional evidence."""
        messages: List[str] = []
        strong = [
            o for o in opinions
            if o.available
            and o.direction in (Direction.UP, Direction.DOWN)
            and o.strength >= _STRONG_THRESHOLD
        ]
        for i in range(len(strong)):
            for j in range(i + 1, len(strong)):
                a, b = strong[i], strong[j]
                if a.direction != b.direction:
                    messages.append(
                        f"{a.source} indicates {a.direction.value} while "
                        f"{b.source} indicates {b.direction.value}"
                    )
        # The base decision itself may be contradicted by strong evidence.
        if base in (Decision.UP, Decision.DOWN):
            base_dir = Direction.UP if base == Decision.UP else Direction.DOWN
            for opinion in strong:
                if opinion.direction != base_dir:
                    messages.append(
                        f"base decision {base.value} conflicts with {opinion.source} "
                        f"indicating {opinion.direction.value}"
                    )
        return _dedupe(messages)

    @staticmethod
    def _weak_evidence(opinions: Sequence[_Opinion]) -> List[str]:
        """Flag unavailable or low-strength evidence sources."""
        weak: List[str] = []
        for opinion in opinions:
            if not opinion.available:
                weak.append(f"{opinion.source} evidence is unavailable")
            elif opinion.strength < _WEAK_THRESHOLD:
                weak.append(
                    f"{opinion.source} evidence is weak (strength {opinion.strength:.2f})"
                )
        if not opinions:
            weak.append("no directional evidence sources were provided")
        return _dedupe(weak)


    @staticmethod
    def _abnormal_conditions(ctx: Mapping[str, Any]) -> List[str]:
        """Detect out-of-distribution or degraded operating conditions."""
        abnormal: List[str] = []
        vol = _as_float(ctx.get("volatility"))
        band = ctx.get("volatility_band")
        if vol is not None and isinstance(band, (list, tuple)) and len(band) == 2:
            low, high = _as_float(band[0]), _as_float(band[1])
            if low is not None and high is not None and not (low <= vol <= high):
                abnormal.append(
                    f"volatility {vol:.4g} is outside the historical band "
                    f"[{low:.4g}, {high:.4g}]"
                )
        elif vol is not None:
            hist = _as_float(ctx.get("historical_volatility"))
            if hist and hist > 0:
                ratio = vol / hist
                if ratio > 2.0:
                    abnormal.append(
                        f"volatility {vol:.4g} is {ratio:.1f}x the historical level"
                    )
                elif ratio < 0.5:
                    abnormal.append(
                        f"volatility {vol:.4g} is only {ratio:.1f}x the historical level"
                    )

        coverage = _as_float(ctx.get("feature_coverage"))
        if coverage is not None and coverage < _MIN_COVERAGE:
            abnormal.append(
                f"feature coverage {coverage:.2f} is below the minimum {_MIN_COVERAGE:.2f}"
            )

        if ctx.get("model_degraded") or ctx.get("vision_degraded") or ctx.get("degraded"):
            abnormal.append(
                "one or more evidence providers are degraded; reliability is reduced"
            )

        data_quality = ctx.get("data_quality")
        if isinstance(data_quality, str) and data_quality.strip().upper() not in ("", "OK"):
            abnormal.append(f"data quality is {data_quality.strip()}")

        return _dedupe(abnormal)

    @staticmethod
    def _invalidating_conditions(ctx: Mapping[str, Any]) -> List[str]:
        """Collect explicit invalidating conditions from the evidence."""
        invalidating = list(_str_list(ctx.get("invalidating_conditions")))
        for key in ("vision", "regime", "technical"):
            mapping = ctx.get(key)
            if isinstance(mapping, Mapping):
                invalidating += _str_list(mapping.get("invalidating_conditions"))
        return _dedupe(invalidating)

    @staticmethod
    def _confidence_adjustment(
        contradictions: Sequence[str],
        weak: Sequence[str],
        abnormal: Sequence[str],
    ) -> float:
        """Compute a non-positive confidence delta from the findings."""
        penalty = (
            0.25 * len(contradictions)
            + 0.10 * len(weak)
            + 0.15 * len(abnormal)
        )
        return -clamp(penalty, 0.0, 1.0)

    def _recommendation(
        self,
        base: Optional[Decision],
        contradictions: Sequence[str],
        ctx: Mapping[str, Any],
        abnormal: Sequence[str],
    ) -> Decision:
        """Decide the final recommendation - only ever towards NO-TRADE."""
        if base is None or base == Decision.NO_TRADE:
            return Decision.NO_TRADE
        if contradictions and self.config.reasoning.hard_contradiction_forces_no_trade:
            return Decision.NO_TRADE
        coverage = _as_float(ctx.get("feature_coverage"))
        if coverage is not None and coverage < _SEVERE_COVERAGE:
            return Decision.NO_TRADE
        dq = ctx.get("data_quality")
        if isinstance(dq, str) and dq.strip().upper() == "BAD":
            return Decision.NO_TRADE
        # No upgrade path exists: the supervisor keeps the base decision at most.
        return base

    @staticmethod
    def _explanation(
        base: Optional[Decision],
        recommendation: Decision,
        contradictions: Sequence[str],
        weak: Sequence[str],
        abnormal: Sequence[str],
        adjustment: float,
    ) -> str:
        """Compose a plain-language explanation of the critique."""
        parts: List[str] = []
        base_text = base.value if base else "none"
        parts.append(
            f"Reviewed base decision {base_text}; recommendation {recommendation.value}."
        )
        if contradictions:
            parts.append("Contradictions: " + "; ".join(contradictions) + ".")
        if weak:
            parts.append("Weak evidence: " + "; ".join(weak) + ".")
        if abnormal:
            parts.append("Abnormal conditions: " + "; ".join(abnormal) + ".")
        if not (contradictions or weak or abnormal):
            parts.append("Evidence is internally consistent with no abnormal conditions.")
        parts.append(f"Confidence adjustment {adjustment:+.2f}.")
        return " ".join(parts)

