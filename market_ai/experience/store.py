"""Structured market-experience memory.

An *experience* is a completed (prediction, outcome) pair.  It is the unit
of learning for Phase 11: everything the system knows about its own mistakes
is derived from this table.

The store is deliberately append-only from the pipeline's point of view.  A
completed experience is never mutated; a new model version produces new
experiences.  That makes the training dataset reproducible from history.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..config import Config
from ..types import (
    Decision,
    Direction,
    ErrorCategory,
    Experience,
    Outcome,
    Prediction,
    Regime,
    Timeframe,
)
from ..utils.logging import get_logger

__all__ = ["ExperienceStore", "build_experience", "derive_tags"]

_log = get_logger(__name__)

#: Regimes in which a false breakout is the classic failure mode.
_RANGE_REGIMES = {Regime.RANGE.value, Regime.LOW_VOLATILITY.value}
_VOLATILE_REGIMES = {Regime.HIGH_VOLATILITY.value, Regime.BREAKOUT.value}


def derive_tags(
    prediction: Prediction,
    outcome: Outcome,
    *,
    features: Optional[Mapping[str, float]] = None,
) -> tuple:
    """Attach interpretable failure tags used by the error-learning analytics.

    Tags are intentionally coarse and rule-based: they must be explainable to
    a human, and they must not themselves be a learned model (an LLM or a
    model must never silently rewrite trading rules).
    """
    tags: List[str] = []
    features = features or {}
    regime = prediction.regime.value
    conf = prediction.confidence

    if outcome.error_category in (ErrorCategory.FALSE_UP, ErrorCategory.FALSE_DOWN):
        if regime in _RANGE_REGIMES:
            tags.append("RANGE_NOISE")
        if regime in _VOLATILE_REGIMES:
            tags.append("VOLATILITY_SPIKE")
        if conf >= 0.7:
            tags.append("OVERCONFIDENCE")
        if prediction.agreement < 0.5:
            tags.append("TIMEFRAME_CONFLICT")
        momentum = features.get("momentum.rsi_14")
        if momentum is not None and 45.0 <= momentum <= 55.0:
            tags.append("LOW_MOMENTUM")
        atr_pct = features.get("volatility.atr_percentile")
        if atr_pct is not None and atr_pct >= 0.85:
            tags.append("VOLATILITY_SPIKE")
        if features.get("pa.range_expansion", 0.0) and regime in _RANGE_REGIMES:
            tags.append("FALSE_BREAKOUT")
    elif outcome.error_category is ErrorCategory.NO_TRADE_MISSED:
        tags.append("UNDERCONFIDENCE")
        if conf < 0.35:
            tags.append("MISSED_STRONG_MOVE")
    elif outcome.error_category is ErrorCategory.NO_TRADE_CORRECT:
        tags.append("GOOD_ABSTENTION")

    # Deduplicate while preserving first-seen order for stable reporting.
    seen: Dict[str, None] = {}
    for tag in tags:
        seen.setdefault(tag, None)
    return tuple(seen.keys())


def build_experience(
    prediction: Prediction,
    outcome: Outcome,
    *,
    tags: Optional[Sequence[str]] = None,
    created_at: Optional[int] = None,
) -> Experience:
    """Join a prediction with its outcome into a learnable experience row."""
    resolved_tags = tuple(tags) if tags is not None else derive_tags(prediction, outcome)
    return Experience(
        prediction_id=prediction.prediction_id,
        timestamp=prediction.timestamp,
        asset=prediction.asset,
        timeframe=prediction.timeframe,
        features=dict(prediction.features),
        regime=prediction.regime,
        decision=prediction.decision,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities.as_dict(),
        model_version=prediction.model_version,
        strategy_version=prediction.strategy_version,
        feature_version=prediction.feature_version,
        actual_direction=outcome.actual_direction,
        error_category=outcome.error_category,
        profit_or_loss=outcome.profit_or_loss,
        tags=resolved_tags,
        created_at=created_at if created_at is not None else int(time.time()),
    )


class ExperienceStore:
    """Append-only memory of completed predictions.

    Works with or without a :class:`~market_ai.storage.db.Database`; when no
    database is supplied it keeps an in-memory list, which keeps unit tests
    and the offline demo path dependency-free.
    """

    def __init__(self, db: Any = None, config: Optional[Config] = None) -> None:
        self.db = db
        self.config = config or Config()
        self._memory: List[Experience] = []

    # -- writes ------------------------------------------------------------
    def add(self, experience: Experience) -> None:
        """Record one completed experience (idempotent on prediction_id)."""
        if self.db is not None:
            self.db.experiences.add(experience)
            return
        for idx, existing in enumerate(self._memory):
            if existing.prediction_id == experience.prediction_id:
                self._memory[idx] = experience
                return
        self._memory.append(experience)

    def add_many(self, items: Iterable[Experience]) -> int:
        count = 0
        for item in items:
            self.add(item)
            count += 1
        return count

    def add_from_prediction(
        self, prediction: Prediction, outcome: Outcome, *, tags: Optional[Sequence[str]] = None
    ) -> Experience:
        """Convenience: build and store an experience in one call."""
        experience = build_experience(prediction, outcome, tags=tags)
        self.add(experience)
        return experience

    # -- reads -------------------------------------------------------------
    def all(self) -> List[Experience]:
        """Every stored experience, oldest first."""
        if self.db is not None:
            rows = self.db.experiences.list(order_by="timestamp")
            return [Experience.from_dict(r) for r in rows]
        return sorted(self._memory, key=lambda e: e.timestamp)

    def count(self) -> int:
        if self.db is not None:
            return self.db.experiences.count()
        return len(self._memory)

    def since(self, ts: int) -> List[Experience]:
        """Experiences recorded at or after ``ts`` (drives retraining triggers)."""
        return [e for e in self.all() if e.timestamp >= ts]

    def since_count(self, ts: int) -> int:
        return len(self.since(ts))

    def latest(self, limit: int = 50) -> List[Experience]:
        return self.all()[-int(limit):]

    def by_regime(self) -> Dict[str, List[Experience]]:
        out: Dict[str, List[Experience]] = {}
        for experience in self.all():
            out.setdefault(experience.regime.value, []).append(experience)
        return out

    def tags(self) -> Dict[str, int]:
        """Frequency of every error tag seen so far."""
        counts: Dict[str, int] = {}
        for experience in self.all():
            for tag in experience.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> Dict[str, Any]:
        """Compact health view for the dashboard."""
        items = self.all()
        if not items:
            return {"count": 0, "regimes": {}, "tags": {}, "first_ts": None, "last_ts": None}
        return {
            "count": len(items),
            "first_ts": items[0].timestamp,
            "last_ts": items[-1].timestamp,
            "regimes": {k: len(v) for k, v in self.by_regime().items()},
            "tags": self.tags(),
        }

    def clear(self) -> None:
        """Drop all experiences.  Used by tests only."""
        if self.db is not None:
            self.db.experiences.clear()
        self._memory.clear()
