"""Schema, validation and coercion for chart-vision model responses.

The vision model is *evidence*, never authority.  Its output is therefore
treated as untrusted input: every field is validated against
:data:`VISION_RESPONSE_SCHEMA` before it may influence anything downstream.
Benign deviations that a well-meaning model routinely produces (missing
optional keys, numbers delivered as strings, probabilities nudged past their
range) are *repaired* by :func:`coerce_payload`; genuinely structural problems
(a payload that is not a mapping, a text field delivered as a list, a
probability delivered as prose) raise :class:`ValueError` so the caller can
degrade to the deterministic offline analyser.

Keeping repair and rejection in one place means the orchestrator
(:class:`market_ai.vision.analyser.ChartVisionAnalyser`) never has to reason
about model quirks itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..types import Direction
from ..utils.stats import clamp

__all__ = [
    "VISION_RESPONSE_SCHEMA",
    "DIRECTION_ALIASES",
    "VisionResult",
    "validate_vision_payload",
    "coerce_payload",
]


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
#: The exact response we ask a multimodal model to produce.  It intentionally
#: excludes the bookkeeping fields (``provider``/``degraded``/``raw``) which we
#: attach ourselves - the model never gets to declare its own provenance.
VISION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "title": "VisionResponse",
    "type": "object",
    "description": (
        "Structured chart-vision evidence. Directional bias must be UP, DOWN or "
        "NEUTRAL; probabilities are the model's self-reported estimates in [0, 1]."
    ),
    # Extra keys are tolerated so a chatty model is not rejected outright; only
    # the declared fields are ever consumed.
    "additionalProperties": True,
    "required": [
        "direction_bias",
        "trend",
        "structure",
        "support",
        "resistance",
        "breakout_probability",
        "reversal_probability",
        "momentum",
        "volatility",
        "confidence",
        "invalidating_conditions",
        "reasoning_summary",
    ],
    "properties": {
        "direction_bias": {"type": "string", "enum": ["UP", "DOWN", "NEUTRAL"]},
        "trend": {"type": "string"},
        "structure": {"type": "string"},
        "support": {"type": "array", "items": {"type": "number"}},
        "resistance": {"type": "array", "items": {"type": "number"}},
        "breakout_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reversal_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "momentum": {"type": "string"},
        "volatility": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "invalidating_conditions": {"type": "array", "items": {"type": "string"}},
        "reasoning_summary": {"type": "string"},
    },
}


#: Synonyms a model may emit for a neutral bias.  Mapping them keeps a
#: "SIDEWAYS" answer usable instead of throwing away otherwise good evidence.


# --------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VisionResult:
    """Normalised chart-vision evidence.

    Instances are immutable value objects so a result can be stored, logged and
    compared safely.  Construction normalises the enum, clamps probabilities
    and copies the sequence fields, which means callers can trust the
    invariants without re-checking them.
    """

    direction_bias: Direction
    trend: str
    structure: str
    support: List[float]
    resistance: List[float]
    breakout_probability: float
    reversal_probability: float
    momentum: str
    volatility: str
    confidence: float
    invalidating_conditions: List[str]
    reasoning_summary: str
    provider: str = "unknown"
    degraded: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.direction_bias, Direction):
            object.__setattr__(
                self, "direction_bias", _coerce_direction(self.direction_bias)
            )
        object.__setattr__(self, "support", _clean_levels(self.support))
        object.__setattr__(self, "resistance", _clean_levels(self.resistance))
        object.__setattr__(
            self, "breakout_probability",
            clamp(self.breakout_probability, 0.0, 1.0),
        )
        object.__setattr__(
            self, "reversal_probability",
            clamp(self.reversal_probability, 0.0, 1.0),
        )
        object.__setattr__(self, "confidence", clamp(self.confidence, 0.0, 1.0))
        object.__setattr__(
            self, "invalidating_conditions",
            [str(x) for x in (self.invalidating_conditions or [])],
        )
        object.__setattr__(self, "trend", str(self.trend))
        object.__setattr__(self, "structure", str(self.structure))
        object.__setattr__(self, "momentum", str(self.momentum))
        object.__setattr__(self, "volatility", str(self.volatility))
        object.__setattr__(self, "reasoning_summary", str(self.reasoning_summary))
        object.__setattr__(self, "provider", str(self.provider))
        object.__setattr__(self, "degraded", bool(self.degraded))
        object.__setattr__(self, "raw", dict(self.raw or {}))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict; the declared fields match the schema."""
        return {
            "direction_bias": self.direction_bias.value,
            "trend": self.trend,
            "structure": self.structure,
            "support": list(self.support),
            "resistance": list(self.resistance),
            "breakout_probability": self.breakout_probability,
            "reversal_probability": self.reversal_probability,
            "momentum": self.momentum,
            "volatility": self.volatility,
            "confidence": self.confidence,
            "invalidating_conditions": list(self.invalidating_conditions),
            "reasoning_summary": self.reasoning_summary,
            "provider": self.provider,
            "degraded": self.degraded,
            "raw": dict(self.raw),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisionResult":
        """Build a result from a (possibly sloppy) mapping via coercion."""
        return coerce_payload(payload)

DIRECTION_ALIASES: Dict[str, str] = {
    "up": "UP", "bull": "UP", "bullish": "UP", "long": "UP", "buy": "UP",
    "positive": "UP", "upside": "UP",
    "down": "DOWN", "bear": "DOWN", "bearish": "DOWN", "short": "DOWN",
    "sell": "DOWN", "negative": "DOWN", "downside": "DOWN",
    "neutral": "NEUTRAL", "sideways": "NEUTRAL", "side": "NEUTRAL",
    "flat": "NEUTRAL", "range": "NEUTRAL", "ranging": "NEUTRAL",
    "none": "NEUTRAL", "consolidation": "NEUTRAL", "consolidating": "NEUTRAL",
}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _is_number(value: Any) -> bool:
    """True for a finite real number (``bool`` is explicitly not a number)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def validate_vision_payload(payload: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a raw vision payload against :data:`VISION_RESPONSE_SCHEMA`.

    Returns ``(ok, errors)``.  ``ok`` is ``True`` only when every required
    field is present *and* well-typed; ``errors`` is a list of human-readable
    messages describing each problem found (empty when ``ok`` is ``True``).

    This is deliberately strict: it is the gate that decides whether a model
    response may be trusted without repair.  Coercion of benign deviations is
    the job of :func:`coerce_payload`.
    """
    errors: List[str] = []
    if not isinstance(payload, Mapping):
        return False, ["payload must be a mapping/object"]

    required = VISION_RESPONSE_SCHEMA["required"]
    for key in required:
        if key not in payload:
            errors.append(f"missing required field: {key}")

    if "direction_bias" in payload:
        value = payload["direction_bias"]
        if not isinstance(value, str) or value not in ("UP", "DOWN", "NEUTRAL"):
            errors.append(
                f"direction_bias must be one of UP/DOWN/NEUTRAL, got {value!r}"
            )

    for key in ("trend", "structure", "momentum", "volatility", "reasoning_summary"):
        if key in payload and not isinstance(payload[key], str):
            errors.append(f"{key} must be a string, got {type(payload[key]).__name__}")

    for key in ("support", "resistance"):
        if key in payload:
            value = payload[key]
            if not isinstance(value, (list, tuple)):
                errors.append(f"{key} must be an array, got {type(value).__name__}")
            elif not all(_is_number(v) for v in value):
                errors.append(f"{key} must contain only finite numbers")

    for key in ("breakout_probability", "reversal_probability", "confidence"):
        if key in payload:
            value = payload[key]
            if not _is_number(value):
                errors.append(f"{key} must be a finite number, got {value!r}")
            elif not (0.0 <= float(value) <= 1.0):
                errors.append(f"{key} must be within [0, 1], got {value!r}")

    if "invalidating_conditions" in payload:
        value = payload["invalidating_conditions"]
        if not isinstance(value, (list, tuple)):
            errors.append("invalidating_conditions must be an array")
        elif not all(isinstance(v, str) for v in value):
            errors.append("invalidating_conditions must contain only strings")

    return (len(errors) == 0), errors


# --------------------------------------------------------------------------
# Coercion
# --------------------------------------------------------------------------
def coerce_payload(payload: Mapping[str, Any]) -> VisionResult:
    """Repair benign deviations in a vision payload and build a result.

    Missing optional keys get safe defaults, numeric strings are parsed,
    out-of-range probabilities are clamped, and level lists are normalised.
    Structural violations (a non-mapping payload, a list where text belongs, a
    non-numeric probability) raise :class:`ValueError` so the caller can
    degrade to the offline analyser rather than trust bad evidence.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("vision payload must be a mapping/object")

    return VisionResult(
        direction_bias=_coerce_direction(payload.get("direction_bias")),
        trend=_coerce_text(payload.get("trend"), "unknown", "trend"),
        structure=_coerce_text(payload.get("structure"), "unknown", "structure"),
        support=_coerce_levels(payload.get("support"), "support"),
        resistance=_coerce_levels(payload.get("resistance"), "resistance"),
        breakout_probability=_coerce_probability(
            payload.get("breakout_probability"), 0.5, "breakout_probability"
        ),
        reversal_probability=_coerce_probability(
            payload.get("reversal_probability"), 0.5, "reversal_probability"
        ),
        momentum=_coerce_text(payload.get("momentum"), "unknown", "momentum"),
        volatility=_coerce_text(payload.get("volatility"), "unknown", "volatility"),
        confidence=_coerce_probability(payload.get("confidence"), 0.0, "confidence"),
        invalidating_conditions=_coerce_str_list(
            payload.get("invalidating_conditions"), "invalidating_conditions"
        ),
        reasoning_summary=_coerce_text(
            payload.get("reasoning_summary"), "", "reasoning_summary"
        ),
        provider=str(payload.get("provider", "unknown")),
        degraded=bool(payload.get("degraded", False)),
        raw=dict(payload),
    )



# --------------------------------------------------------------------------
# internal coercion helpers
# --------------------------------------------------------------------------
def _coerce_direction(value: Any) -> Direction:
    """Map a loose directional label onto :class:`Direction` (never crashes)."""
    if value is None:
        return Direction.NEUTRAL
    if isinstance(value, Direction):
        return value
    if isinstance(value, str):
        canonical = DIRECTION_ALIASES.get(value.strip().lower())
        # An unrecognised label is still only a *bias*; collapsing to NEUTRAL
        # is the safe, non-crashing choice (validate() flags it separately).
        return Direction(canonical) if canonical else Direction.NEUTRAL
    raise ValueError(f"direction_bias must be a string, got {type(value).__name__}")


def _coerce_text(value: Any, default: str, field_name: str) -> str:
    """Coerce a text field; ``None`` -> default, numbers -> str, else raise."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if _is_number(value):
        return str(value)  # benign: a numeric label
    raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")


def _coerce_probability(value: Any, default: float, field_name: str) -> float:
    """Coerce a probability, clamping into ``[0, 1]``; raise if not numeric."""
    if value is None:
        return float(default)
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, got bool")
    if isinstance(value, (int, float)):
        fval = float(value)
    elif isinstance(value, str):
        try:
            fval = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} is not numeric: {value!r}") from exc
    else:
        raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}")
    if not math.isfinite(fval):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return clamp(fval, 0.0, 1.0)


def _coerce_levels(value: Any, field_name: str) -> List[float]:
    """Coerce a support/resistance list, dropping anything non-finite."""
    if value is None:
        return []
    if _is_number(value):  # a lone number is a benign single-level list
        return [float(value)]
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be an array, got str")
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            if _is_number(item):
                out.append(float(item))
            elif isinstance(item, str):
                try:
                    fval = float(item.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"{field_name} contains a non-numeric entry: {item!r}"
                    ) from exc
                if math.isfinite(fval):
                    out.append(fval)
            # Non-finite / null entries are dropped rather than invented.
        return out
    raise ValueError(f"{field_name} must be an array, got {type(value).__name__}")


def _coerce_str_list(value: Any, field_name: str) -> List[str]:
    """Coerce a list-of-strings field; a lone string becomes a one-item list."""
    if value is None:
        return []
    if isinstance(value, str):  # a lone string is a benign one-item list
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    raise ValueError(f"{field_name} must be an array, got {type(value).__name__}")


def _clean_levels(values: Sequence[Any]) -> List[float]:
    """Copy a level sequence, keeping only finite numbers."""
    out: List[float] = []
    for item in values or []:
        try:
            fval = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            out.append(fval)
    return out

