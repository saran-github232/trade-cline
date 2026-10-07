"""Core domain types and contracts for the market-analysis system.

Every module in this package depends on the types defined here.  They are
deliberately dependency-free (standard library only) so the system remains
*local-first* and reproducible: no numpy, no pandas, no network.

Design rules encoded in this module
-----------------------------------
1. Time is always stored as **UTC epoch seconds (int)**.  No naive datetimes
   are allowed to leak into the pipeline.  ``to_utc_epoch`` is the only
   sanctioned conversion entry point.
2. ``Candle`` and ``MarketSeries`` are immutable value objects.  Immutability
   is what makes leakage testing meaningful: a slice of history cannot be
   mutated in place by accident.
3. ``NO_TRADE`` is a first-class member of ``Decision``.  It is never a
   fallback bolted on at the UI layer.
4. Probability containers are validated on construction: they must be finite,
   within [0, 1] and sum to 1 within a small tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Timeframe",
    "Direction",
    "Decision",
    "Regime",
    "ErrorCategory",
    "DataQuality",
    "Candle",
    "MarketSeries",
    "FeatureVector",
    "Signal",
    "ProbabilitySet",
    "Prediction",
    "Outcome",
    "Experience",
    "to_utc_epoch",
    "epoch_to_iso",
    "iso_to_epoch",
    "SCHEMA_VERSION",
]

#: Bumped whenever the on-disk record schema changes in a breaking way.
SCHEMA_VERSION = "1.0.0"

EPS = 1e-9


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------
def to_utc_epoch(value: Any, *, assume_tz: str = "UTC") -> int:
    """Normalise any supported timestamp representation to UTC epoch seconds.

    Accepted inputs: ``int``/``float`` epoch seconds, ISO-8601 strings
    (``Z`` suffix or explicit offset), and timezone-aware ``datetime``
    objects.  A naive ``datetime`` is interpreted using ``assume_tz`` which
    defaults to UTC - naive local timestamps are the single most common
    source of silent look-ahead bias, so the default is deliberately strict.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise TypeError("bool is not a valid timestamp")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite timestamp: {value!r}")
        return int(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_resolve_tz(assume_tz))
        return int(value.timestamp())
    if isinstance(value, str):
        return iso_to_epoch(value, assume_tz=assume_tz)
    raise TypeError(f"unsupported timestamp type: {type(value).__name__}")


def _resolve_tz(name: str):
    if name.upper() in ("UTC", "Z", "GMT"):
        return timezone.utc
    if name.upper() in ("IST", "ASIA/KOLKATA"):
        return timezone(timedelta(hours=5, minutes=30))
    raise ValueError(
        f"unsupported timezone {name!r}; use an explicit UTC offset instead"
    )


def iso_to_epoch(value: str, *, assume_tz: str = "UTC") -> int:
    """Parse an ISO-8601 string into UTC epoch seconds."""
    text = value.strip()
    if not text:
        raise ValueError("empty timestamp string")
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unparseable timestamp {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_resolve_tz(assume_tz))
    return int(dt.timestamp())


def epoch_to_iso(epoch: int) -> str:
    """Render UTC epoch seconds as a canonical ISO-8601 string."""
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------
class Timeframe(str, Enum):
    """Supported candle timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"

    @property
    def seconds(self) -> int:
        return _TIMEFRAME_SECONDS[self]

    @property
    def rank(self) -> int:
        """Higher rank == higher (slower) timeframe."""
        return _TIMEFRAME_RANK[self]

    @classmethod
    def parse(cls, value: Any) -> "Timeframe":
        if isinstance(value, Timeframe):
            return value
        text = str(value).strip().lower()
        aliases = {
            "1m": cls.M1, "m1": cls.M1, "60s": cls.M1, "1min": cls.M1,
            "5m": cls.M5, "m5": cls.M5, "300s": cls.M5, "5min": cls.M5,
            "15m": cls.M15, "m15": cls.M15, "900s": cls.M15, "15min": cls.M15,
            "1h": cls.H1, "h1": cls.H1, "60m": cls.H1, "3600s": cls.H1,
        }
        if text not in aliases:
            raise ValueError(
                f"unsupported timeframe {value!r}; expected one of "
                f"{sorted(t.value for t in cls)}"
            )
        return aliases[text]


_TIMEFRAME_SECONDS: Dict[Timeframe, int] = {
    Timeframe.M1: 60,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.H1: 3600,
}
_TIMEFRAME_RANK: Dict[Timeframe, int] = {
    Timeframe.M1: 0,
    Timeframe.M5: 1,
    Timeframe.M15: 2,
    Timeframe.H1: 3,
}


class Direction(str, Enum):
    """Directional bias.  NEUTRAL is a legitimate, frequently correct answer."""

    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"

    @property
    def sign(self) -> int:
        return {Direction.UP: 1, Direction.DOWN: -1, Direction.NEUTRAL: 0}[self]


class Decision(str, Enum):
    """The only three outputs the system may ever produce.

    ``NO_TRADE`` is first-class: it is the *default* when evidence is weak,
    conflicting, incomplete, or out of distribution.
    """

    UP = "UP"
    DOWN = "DOWN"
    NO_TRADE = "NO-TRADE"


class Regime(str, Enum):
    """Market regime labels produced by the regime engine."""

    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNCERTAIN = "UNCERTAIN"

    @property
    def is_trend(self) -> bool:
        return self in (
            Regime.STRONG_UPTREND,
            Regime.WEAK_UPTREND,
            Regime.STRONG_DOWNTREND,
            Regime.WEAK_DOWNTREND,
        )


class ErrorCategory(str, Enum):
    """How a completed prediction turned out - used for error learning."""

    CORRECT_UP = "CORRECT_UP"
    CORRECT_DOWN = "CORRECT_DOWN"
    FALSE_UP = "FALSE_UP"
    FALSE_DOWN = "FALSE_DOWN"
    NO_TRADE_CORRECT = "NO_TRADE_CORRECT"
    NO_TRADE_MISSED = "NO_TRADE_MISSED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_correct(self) -> bool:
        return self in (
            ErrorCategory.CORRECT_UP,
            ErrorCategory.CORRECT_DOWN,
            ErrorCategory.NO_TRADE_CORRECT,
        )


class DataQuality(str, Enum):
    """Coarse data-quality classification consumed by the decision gate."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    BAD = "BAD"

    @property
    def score(self) -> float:
        return {DataQuality.OK: 1.0, DataQuality.DEGRADED: 0.5, DataQuality.BAD: 0.0}[self]



# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Candle:
    """A single immutable OHLC(V) bar.

    ``timestamp`` is the **open time** of the bar in UTC epoch seconds, which
    is the convention used by essentially every exchange.  The bar is only
    *complete* at ``timestamp + timeframe.seconds``; the replay layer is
    responsible for enforcing that.
    """

    timestamp: int
    asset: str
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.timeframe, Timeframe):
            object.__setattr__(self, "timeframe", Timeframe.parse(self.timeframe))
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"candle {name} must be a finite number, got {value!r}")
        vol = self.volume
        if vol is None:
            object.__setattr__(self, "volume", 0.0)
        elif not isinstance(vol, (int, float)) or not math.isfinite(float(vol)):
            raise ValueError(f"candle volume must be finite, got {vol!r}")
        object.__setattr__(self, "timestamp", int(self.timestamp))

    # -- derived geometry ---------------------------------------------------
    @property
    def close_time(self) -> int:
        """UTC epoch second at which this candle becomes final."""
        return self.timestamp + self.timeframe.seconds

    @property
    def body(self) -> float:
        return self.close - self.open

    @property
    def body_abs(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(0.0, self.high - self.low)

    @property
    def upper_wick(self) -> float:
        return max(0.0, self.high - max(self.open, self.close))

    @property
    def lower_wick(self) -> float:
        return max(0.0, min(self.open, self.close) - self.low)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        rng = self.range
        return rng > 0 and self.body_abs / rng <= 0.1

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    def is_valid(self) -> bool:
        """OHLC sanity: high must bound, low must floor, range non-negative."""
        return (
            self.high >= self.low - EPS
            and self.high >= max(self.open, self.close) - EPS
            and self.low <= min(self.open, self.close) + EPS
            and self.range >= 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "asset": self.asset,
            "timeframe": self.timeframe.value,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Candle":
        return cls(
            timestamp=to_utc_epoch(payload["timestamp"]),
            asset=str(payload.get("asset", "")),
            timeframe=Timeframe.parse(payload["timeframe"]),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload.get("volume", 0.0) or 0.0),
        )



@dataclass(frozen=True)
class MarketSeries:
    """An ordered, gap-aware series of candles for one asset/timeframe."""

    asset: str
    timeframe: Timeframe
    candles: Tuple[Candle, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candles", tuple(self.candles))
        if not isinstance(self.timeframe, Timeframe):
            object.__setattr__(self, "timeframe", Timeframe.parse(self.timeframe))

    def __len__(self) -> int:
        return len(self.candles)

    def __iter__(self):
        return iter(self.candles)

    def __getitem__(self, index):
        return self.candles[index]

    def __bool__(self) -> bool:
        return bool(self.candles)

    @property
    def first_timestamp(self) -> Optional[int]:
        return self.candles[0].timestamp if self.candles else None

    @property
    def last_timestamp(self) -> Optional[int]:
        return self.candles[-1].timestamp if self.candles else None

    @property
    def closes(self) -> List[float]:
        return [c.close for c in self.candles]

    @property
    def highs(self) -> List[float]:
        return [c.high for c in self.candles]

    @property
    def lows(self) -> List[float]:
        return [c.low for c in self.candles]

    @property
    def opens(self) -> List[float]:
        return [c.open for c in self.candles]

    @property
    def volumes(self) -> List[float]:
        return [c.volume for c in self.candles]

    def is_sorted(self) -> bool:
        return all(
            self.candles[i].timestamp < self.candles[i + 1].timestamp
            for i in range(len(self.candles) - 1)
        )

    def append(self, candle: Candle) -> "MarketSeries":
        return replace(self, candles=self.candles + (candle,))

    def slice(self, start: int = 0, end: Optional[int] = None) -> "MarketSeries":
        return replace(self, candles=self.candles[start:end])

    def visible_at(self, as_of: int, *, strict: bool = True) -> "MarketSeries":
        """Return only candles whose **close time** is <= ``as_of``.

        This is *the* leakage firewall of the system.  A candle that opened
        before ``as_of`` but closes after it is not yet knowable and is
        therefore excluded.  ``strict=False`` additionally requires the close
        time to be strictly less than ``as_of``.
        """
        limit = as_of if strict else as_of - 1
        kept = tuple(c for c in self.candles if c.close_time <= limit)
        return replace(self, candles=kept)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.candles]

    @classmethod
    def from_dicts(
        cls, asset: str, timeframe: Any, rows: Iterable[Mapping[str, Any]]
    ) -> "MarketSeries":
        tf = Timeframe.parse(timeframe)
        return cls(asset=asset, timeframe=tf, candles=tuple(Candle.from_dict(r) for r in rows))



# --------------------------------------------------------------------------
# Features and signals
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureVector:
    """A versioned, timestamped, causality-safe bundle of numeric features."""

    timestamp: int
    asset: str
    timeframe: Timeframe
    version: str
    values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned: Dict[str, float] = {}
        for key, value in self.values.items():
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            cleaned[str(key)] = fval if math.isfinite(fval) else float("nan")
        object.__setattr__(self, "values", cleaned)
        if not isinstance(self.timeframe, Timeframe):
            object.__setattr__(self, "timeframe", Timeframe.parse(self.timeframe))

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.values.get(key, default)

    @property
    def names(self) -> List[str]:
        return sorted(self.values.keys())

    def vector(self, names: Sequence[str]) -> List[float]:
        """Dense vector in a fixed feature order (NaN where missing)."""
        return [self.values.get(n, float("nan")) for n in names]

    def finite_count(self) -> int:
        return sum(1 for v in self.values.values() if math.isfinite(v))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "asset": self.asset,
            "timeframe": self.timeframe.value,
            "version": self.version,
            "values": dict(self.values),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureVector":
        return cls(
            timestamp=to_utc_epoch(payload["timestamp"]),
            asset=str(payload.get("asset", "")),
            timeframe=Timeframe.parse(payload["timeframe"]),
            version=str(payload.get("version", "unknown")),
            values=dict(payload.get("values", {})),
        )


@dataclass(frozen=True)
class Signal:
    """One evidence provider's opinion at a point in time.

    ``probability`` is the provider's confidence that price moves UP over the
    horizon (0..1).  ``confidence`` is how much the provider trusts *itself*
    (0..1) - these are different things and must not be conflated.
    """

    source: str
    direction: Direction
    probability: float
    confidence: float
    timestamp: int
    evidence: Mapping[str, Any] = field(default_factory=dict)
    available: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            object.__setattr__(self, "direction", Direction(str(self.direction)))
        object.__setattr__(self, "probability", _clamp01(self.probability, "probability"))
        object.__setattr__(self, "confidence", _clamp01(self.confidence, "confidence"))

    @property
    def probability_down(self) -> float:
        return 1.0 - self.probability

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "direction": self.direction.value,
            "probability": self.probability,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "evidence": dict(self.evidence),
            "available": self.available,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Signal":
        return cls(
            source=str(payload["source"]),
            direction=Direction(str(payload["direction"])),
            probability=float(payload.get("probability", 0.5)),
            confidence=float(payload.get("confidence", 0.0)),
            timestamp=to_utc_epoch(payload["timestamp"]),
            evidence=dict(payload.get("evidence", {})),
            available=bool(payload.get("available", True)),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class ProbabilitySet:
    """Normalised P(UP), P(DOWN), P(NO_TRADE) triple."""

    p_up: float
    p_down: float
    p_no_trade: float

    def __post_init__(self) -> None:
        total = float(self.p_up) + float(self.p_down) + float(self.p_no_trade)
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"invalid probability set total={total!r}")
        scale = 1.0 / total
        object.__setattr__(self, "p_up", float(self.p_up) * scale)
        object.__setattr__(self, "p_down", float(self.p_down) * scale)
        object.__setattr__(self, "p_no_trade", float(self.p_no_trade) * scale)

    @property
    def directional_mass(self) -> float:
        return self.p_up + self.p_down

    def as_dict(self) -> Dict[str, float]:
        return {
            "p_up": round(self.p_up, 6),
            "p_down": round(self.p_down, 6),
            "p_no_trade": round(self.p_no_trade, 6),
        }

    def best(self) -> Decision:
        if self.p_no_trade >= self.p_up and self.p_no_trade >= self.p_down:
            return Decision.NO_TRADE
        return Decision.UP if self.p_up >= self.p_down else Decision.DOWN



# --------------------------------------------------------------------------
# Predictions, outcomes, experiences
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Prediction:
    """A fully-specified, fully-auditable prediction record."""

    prediction_id: str
    timestamp: int
    asset: str
    timeframe: Timeframe
    decision: Decision
    probabilities: ProbabilitySet
    confidence: float
    regime: Regime
    agreement: float
    reason: str
    model_version: str
    dataset_version: str
    feature_version: str
    strategy_version: str
    horizon_seconds: int
    entry_price: float
    expires_at: int
    signals: Tuple[Signal, ...] = ()
    features: Mapping[str, float] = field(default_factory=dict)
    vision: Mapping[str, Any] = field(default_factory=dict)
    reasoning: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "confidence", _clamp01(self.confidence, "confidence"))
        object.__setattr__(self, "agreement", _clamp01(self.agreement, "agreement"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp,
            "asset": self.asset,
            "timeframe": self.timeframe.value,
            "decision": self.decision.value,
            "probabilities": self.probabilities.as_dict(),
            "confidence": self.confidence,
            "regime": self.regime.value,
            "agreement": self.agreement,
            "reason": self.reason,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "strategy_version": self.strategy_version,
            "horizon_seconds": self.horizon_seconds,
            "entry_price": self.entry_price,
            "expires_at": self.expires_at,
            "signals": [s.to_dict() for s in self.signals],
            "features": dict(self.features),
            "vision": dict(self.vision),
            "reasoning": dict(self.reasoning),
            "diagnostics": dict(self.diagnostics),
            "schema_version": self.schema_version,
        }


    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Prediction":
        probs = payload["probabilities"]
        return cls(
            prediction_id=str(payload["prediction_id"]),
            timestamp=to_utc_epoch(payload["timestamp"]),
            asset=str(payload["asset"]),
            timeframe=Timeframe.parse(payload["timeframe"]),
            decision=Decision(str(payload["decision"])),
            probabilities=ProbabilitySet(
                float(probs["p_up"]), float(probs["p_down"]), float(probs["p_no_trade"])
            ),
            confidence=float(payload.get("confidence", 0.0)),
            regime=Regime(str(payload.get("regime", Regime.UNCERTAIN.value))),
            agreement=float(payload.get("agreement", 0.0)),
            reason=str(payload.get("reason", "")),
            model_version=str(payload.get("model_version", "unknown")),
            dataset_version=str(payload.get("dataset_version", "unknown")),
            feature_version=str(payload.get("feature_version", "unknown")),
            strategy_version=str(payload.get("strategy_version", "unknown")),
            horizon_seconds=int(payload.get("horizon_seconds", 0)),
            entry_price=float(payload.get("entry_price", 0.0)),
            expires_at=int(payload.get("expires_at", 0)),
            signals=tuple(Signal.from_dict(s) for s in payload.get("signals", [])),
            features=dict(payload.get("features", {})),
            vision=dict(payload.get("vision", {})),
            reasoning=dict(payload.get("reasoning", {})),
            diagnostics=dict(payload.get("diagnostics", {})),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class Outcome:
    """The realised result of a prediction once its horizon has elapsed."""

    prediction_id: str
    resolved_at: int
    actual_direction: Direction
    exit_price: float
    error_category: ErrorCategory
    profit_or_loss: float = 0.0
    payout: float = 0.0
    note: str = ""

    @property
    def is_win(self) -> bool:
        return self.error_category in (
            ErrorCategory.CORRECT_UP,
            ErrorCategory.CORRECT_DOWN,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "resolved_at": self.resolved_at,
            "actual_direction": self.actual_direction.value,
            "exit_price": self.exit_price,
            "error_category": self.error_category.value,
            "profit_or_loss": self.profit_or_loss,
            "payout": self.payout,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Outcome":
        return cls(
            prediction_id=str(payload["prediction_id"]),
            resolved_at=to_utc_epoch(payload["resolved_at"]),
            actual_direction=Direction(str(payload["actual_direction"])),
            exit_price=float(payload.get("exit_price", 0.0)),
            error_category=ErrorCategory(str(payload["error_category"])),
            profit_or_loss=float(payload.get("profit_or_loss", 0.0)),
            payout=float(payload.get("payout", 0.0)),
            note=str(payload.get("note", "")),
        )




@dataclass(frozen=True)
class Experience:
    """A completed (prediction, outcome) pair - the unit of learning."""

    prediction_id: str
    timestamp: int
    asset: str
    timeframe: Timeframe
    features: Mapping[str, float]
    regime: Regime
    decision: Decision
    confidence: float
    probabilities: Mapping[str, float]
    model_version: str
    strategy_version: str
    feature_version: str
    actual_direction: Direction
    error_category: ErrorCategory
    profit_or_loss: float
    tags: Tuple[str, ...] = ()
    created_at: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp,
            "asset": self.asset,
            "timeframe": self.timeframe.value,
            "features": dict(self.features),
            "regime": self.regime.value,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "model_version": self.model_version,
            "strategy_version": self.strategy_version,
            "feature_version": self.feature_version,
            "actual_direction": self.actual_direction.value,
            "error_category": self.error_category.value,
            "profit_or_loss": self.profit_or_loss,
            "tags": list(self.tags),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Experience":
        return cls(
            prediction_id=str(payload["prediction_id"]),
            timestamp=to_utc_epoch(payload["timestamp"]),
            asset=str(payload["asset"]),
            timeframe=Timeframe.parse(payload["timeframe"]),
            features=dict(payload.get("features", {})),
            regime=Regime(str(payload.get("regime", Regime.UNCERTAIN.value))),
            decision=Decision(str(payload["decision"])),
            confidence=float(payload.get("confidence", 0.0)),
            probabilities=dict(payload.get("probabilities", {})),
            model_version=str(payload.get("model_version", "unknown")),
            strategy_version=str(payload.get("strategy_version", "unknown")),
            feature_version=str(payload.get("feature_version", "unknown")),
            actual_direction=Direction(str(payload["actual_direction"])),
            error_category=ErrorCategory(str(payload["error_category"])),
            profit_or_loss=float(payload.get("profit_or_loss", 0.0)),
            tags=tuple(payload.get("tags", ())),
            created_at=int(payload.get("created_at", 0)),
        )


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------
def _clamp01(value: float, name: str) -> float:
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return 0.0 if fval < 0.0 else (1.0 if fval > 1.0 else fval)

