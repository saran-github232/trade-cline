"""Schema, validation and coercion for reasoning-supervisor responses.

The reasoning supervisor is a *safety* component.  It may only ever reduce
confidence or push a decision towards NO-TRADE; it may never invent market
data, change weights/risk limits, or create a directional call.  The schema
and coercion here enforce the shape of an (optional) external model response,
while :mod:`market_ai.reasoning.supervisor` enforces the *semantics*.

Two invariants are baked into :class:`ReasoningResult` construction:

* ``confidence_adjustment`` is clamped into ``[-1, 0]`` - it can only reduce;
* ``recommendation`` is always a :class:`Decision`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..types import Decision
from ..utils.stats import clamp

__all__ = [
    "REASONING_RESPONSE_SCHEMA",
    "DECISION_ALIASES",
    "ReasoningResult",
    "validate_reasoning_payload",
    "coerce_reasoning_payload",
]


#: The structured response we ask an (optional) external supervisor to produce.
REASONING_RESPONSE_SCHEMA: Dict[str, Any] = {
    "title": "ReasoningResponse",
    "type": "object",
    "description": (
        "Supervisor critique. recommendation may only be NO-TRADE or the base "
        "decision; confidence_adjustment is a non-positive confidence delta."
    ),
    "additionalProperties": True,
    "required": [
        "recommendation",
        "contradictions",
        "weak_evidence",
        "invalidating_conditions",
        "abnormal_conditions",
        "explanation",
        "confidence_adjustment",
    ],
    "properties": {
        "recommendation": {"type": "string", "enum": ["UP", "DOWN", "NO-TRADE"]},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "weak_evidence": {"type": "array", "items": {"type": "string"}},
        "invalidating_conditions": {"type": "array", "items": {"type": "string"}},
        "abnormal_conditions": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
        "confidence_adjustment": {"type": "number", "minimum": -1.0, "maximum": 0.0},
    },
}


#: Synonyms accepted for a decision label.
DECISION_ALIASES: Dict[str, str] = {
    "up": "UP", "long": "UP", "bull": "UP", "bullish": "UP", "buy": "UP",
    "down": "DOWN", "short": "DOWN", "bear": "DOWN", "bearish": "DOWN", "sell": "DOWN",
    "no-trade": "NO-TRADE", "no_trade": "NO-TRADE", "notrade": "NO-TRADE",
    "no trade": "NO-TRADE", "none": "NO-TRADE", "neutral": "NO-TRADE",
    "skip": "NO-TRADE", "flat": "NO-TRADE", "hold": "NO-TRADE",
}


@dataclass(frozen=True)
class ReasoningResult:
    """The supervisor's critique of a candidate decision.

    ``confidence_adjustment`` is always in ``[-1, 0]`` and ``recommendation``
    is always a :class:`Decision`; both are enforced on construction so no
    caller can accidentally widen the supervisor's authority.
    """

    recommendation: Decision
    contradictions: List[str]
    weak_evidence: List[str]
    invalidating_conditions: List[str]
    abnormal_conditions: List[str]
    explanation: str
    confidence_adjustment: float
    provider: str = "offline"
    degraded: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.recommendation, Decision):
            object.__setattr__(
                self, "recommendation", _coerce_decision(self.recommendation)
            )
        # The supervisor may ONLY reduce confidence: clamp to [-1, 0].
        object.__setattr__(
            self, "confidence_adjustment",
            -clamp(abs(float(self.confidence_adjustment)), 0.0, 1.0),
        )
        object.__setattr__(self, "contradictions", _str_list(self.contradictions))
        object.__setattr__(self, "weak_evidence", _str_list(self.weak_evidence))
        object.__setattr__(
            self, "invalidating_conditions", _str_list(self.invalidating_conditions)
        )
        object.__setattr__(
            self, "abnormal_conditions", _str_list(self.abnormal_conditions)
        )
        object.__setattr__(self, "explanation", str(self.explanation))
        object.__setattr__(self, "provider", str(self.provider))
        object.__setattr__(self, "degraded", bool(self.degraded))
        object.__setattr__(self, "raw", dict(self.raw or {}))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "recommendation": self.recommendation.value,
            "contradictions": list(self.contradictions),
            "weak_evidence": list(self.weak_evidence),
            "invalidating_conditions": list(self.invalidating_conditions),
            "abnormal_conditions": list(self.abnormal_conditions),
            "explanation": self.explanation,
            "confidence_adjustment": self.confidence_adjustment,
            "provider": self.provider,
            "degraded": self.degraded,
            "raw": dict(self.raw),
        }



def validate_reasoning_payload(payload: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a raw supervisor payload; return ``(ok, errors)``."""
    errors: List[str] = []
    if not isinstance(payload, Mapping):
        return False, ["payload must be a mapping/object"]

    for key in REASONING_RESPONSE_SCHEMA["required"]:
        if key not in payload:
            errors.append(f"missing required field: {key}")

    if "recommendation" in payload:
        value = payload["recommendation"]
        if not isinstance(value, str) or value not in ("UP", "DOWN", "NO-TRADE"):
            errors.append(f"recommendation must be UP/DOWN/NO-TRADE, got {value!r}")

    for key in (
        "contradictions",
        "weak_evidence",
        "invalidating_conditions",
        "abnormal_conditions",
    ):
        if key in payload:
            value = payload[key]
            if not isinstance(value, (list, tuple)):
                errors.append(f"{key} must be an array")
            elif not all(isinstance(v, str) for v in value):
                errors.append(f"{key} must contain only strings")

    if "explanation" in payload and not isinstance(payload["explanation"], str):
        errors.append("explanation must be a string")

    if "confidence_adjustment" in payload:
        value = payload["confidence_adjustment"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            errors.append(f"confidence_adjustment must be a finite number, got {value!r}")
        elif not (-1.0 <= float(value) <= 0.0):
            errors.append(
                f"confidence_adjustment must be within [-1, 0], got {value!r}"
            )

    return (len(errors) == 0), errors


def coerce_reasoning_payload(payload: Mapping[str, Any]) -> ReasoningResult:
    """Repair benign deviations in a supervisor payload and build a result.

    Structural violations raise :class:`ValueError`; the caller degrades to the
    offline supervisor rather than trusting bad output.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("reasoning payload must be a mapping/object")
    return ReasoningResult(
        recommendation=_coerce_decision(payload.get("recommendation")),
        contradictions=_coerce_str_list(payload.get("contradictions"), "contradictions"),
        weak_evidence=_coerce_str_list(payload.get("weak_evidence"), "weak_evidence"),
        invalidating_conditions=_coerce_str_list(
            payload.get("invalidating_conditions"), "invalidating_conditions"
        ),
        abnormal_conditions=_coerce_str_list(
            payload.get("abnormal_conditions"), "abnormal_conditions"
        ),
        explanation=_coerce_text(payload.get("explanation")),
        confidence_adjustment=_coerce_adjustment(payload.get("confidence_adjustment")),
        provider=str(payload.get("provider", "unknown")),
        degraded=bool(payload.get("degraded", False)),
        raw=dict(payload),
    )



# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------
def _coerce_decision(value: Any) -> Decision:
    """Map a loose decision label onto :class:`Decision` (defaults to NO-TRADE)."""
    if value is None:
        return Decision.NO_TRADE
    if isinstance(value, Decision):
        return value
    if isinstance(value, str):
        canonical = DECISION_ALIASES.get(value.strip().lower())
        if canonical:
            return Decision(canonical)
        # Unknown labels must never become a directional call -> NO-TRADE.
        return Decision.NO_TRADE
    raise ValueError(f"recommendation must be a string, got {type(value).__name__}")


def _coerce_adjustment(value: Any) -> float:
    """Coerce a confidence delta, clamping into ``[-1, 0]``."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        raise ValueError("confidence_adjustment must be numeric, got bool")
    if isinstance(value, (int, float)):
        fval = float(value)
    elif isinstance(value, str):
        try:
            fval = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"confidence_adjustment is not numeric: {value!r}") from exc
    else:
        raise ValueError(
            f"confidence_adjustment must be numeric, got {type(value).__name__}"
        )
    if not math.isfinite(fval):
        raise ValueError(f"confidence_adjustment must be finite, got {value!r}")
    return -clamp(abs(fval), 0.0, 1.0)


def _coerce_text(value: Any) -> str:
    """Coerce an explanation field to a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise ValueError(f"explanation must be a string, got {type(value).__name__}")


def _coerce_str_list(value: Any, field_name: str) -> List[str]:
    """Coerce a list-of-strings field; a lone string becomes a one-item list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    raise ValueError(f"{field_name} must be an array, got {type(value).__name__}")


def _str_list(values: Sequence[Any]) -> List[str]:
    """Copy a sequence into a list of strings."""
    return [str(v) for v in (values or []) if v is not None]

