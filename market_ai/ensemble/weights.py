"""Validation-based ensemble weighting.

The weights in :class:`~market_ai.config.EnsembleConfig` are **initialisation
values only**.  This module measures whether different weight vectors actually
generalise better on data the system has never seen, which is the only
justification for changing them.

Method
------
Given validation records of the form ``(signals, actual_direction)`` we score
candidate weight vectors with two complementary objectives:

* **Brier score** on P(UP) - rewards honest probabilities everywhere, so the
  optimiser cannot win by only being right on a handful of loud calls.
* **Directional precision at the gate** - rewards weights that produce
tradeable decisions which are actually correct.

A candidate must beat the incumbent on a *held-out* split before it is
returned, and the search is deliberately coarse (coordinate descent over a
small grid).  Fine-grained weight search on a few hundred samples is exactly
the parameter over-optimisation the audit is meant to catch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..types import Direction, ProbabilitySet, Signal
from ..utils.logging import get_logger
from ..utils.stats import brier_score, clamp, safe_div
from .engine import WEIGHTED_PROVIDERS, EnsembleEngine

__all__ = ["WeightCandidate", "ValidationWeightOptimizer", "evaluate_weights"]

_log = get_logger(__name__)


@dataclass
class WeightCandidate:
    """A scored weight vector."""

    weights: Dict[str, float]
    brier: float
    precision: float
    n_trades: int
    n_samples: int
    score: float = 0.0
    accepted: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": {k: round(v, 6) for k, v in self.weights.items()},
            "brier": None if self.brier != self.brier else round(self.brier, 6),
            "precision": None if self.precision != self.precision else round(self.precision, 6),
            "n_trades": self.n_trades,
            "n_samples": self.n_samples,
            "score": round(self.score, 6),
            "accepted": self.accepted,
            "reason": self.reason,
        }


def evaluate_weights(
    records: Sequence[Tuple[Sequence[Signal], Direction]],
    weights: Mapping[str, float],
    *,
    config: Optional[Config] = None,
    min_trades: int = 20,
) -> WeightCandidate:
    """Score one weight vector against labelled validation records.

    Records whose outcome is NEUTRAL are excluded from the precision term but
    still contribute to the Brier score, because a neutral bar is genuinely
    ambiguous rather than a win or a loss.
    """
    cfg = config or Config()
    engine = EnsembleEngine(cfg, weights=weights)

    probs: List[float] = []
    outcomes: List[int] = []
    trade_correct = 0
    trade_total = 0

    for signals, actual in records:
        result = engine.combine(list(signals), timestamp=0)
        p = result.probabilities
        if actual in (Direction.UP, Direction.DOWN):
            probs.append(p.p_up)
            outcomes.append(1 if actual is Direction.UP else 0)
        # A trade only happens when the gate would plausibly allow it: the
        # dominant direction must clear the configured confidence floor.
        leading = max(p.p_up, p.p_down)
        if leading > cfg.decision.min_confidence and leading > p.p_no_trade:
            trade_total += 1
            if actual is Direction.UP and p.p_up > p.p_down:
                trade_correct += 1
            elif actual is Direction.DOWN and p.p_down > p.p_up:
                trade_correct += 1

    brier = brier_score(probs, outcomes) if probs else float("nan")
    precision = safe_div(trade_correct, trade_total, float("nan"))

    # Composite score.  Brier is bounded in [0, 1] so a (1 - brier) term is
    # comparable to precision.  Precision is only counted once the sample is
    # large enough to mean anything - otherwise the optimiser would chase a
    # 3-trade 100% fluke.
    brier_term = (1.0 - brier) if brier == brier else 0.0
    precision_term = precision if (precision == precision and trade_total >= min_trades) else brier_term
    score = 0.5 * brier_term + 0.5 * precision_term

    return WeightCandidate(
        weights=dict(weights),
        brier=brier,
        precision=precision,
        n_trades=trade_total,
        n_samples=len(records),
        score=score,
    )


class ValidationWeightOptimizer:
    """Coarse coordinate-descent search for better ensemble weights."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        step: float = 0.05,
        min_weight: float = 0.0,
        max_weight: float = 0.40,
        passes: int = 2,
    ) -> None:
        self.config = config or Config()
        self.step = step
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.passes = passes

    def optimize(
        self,
        train_records: Sequence[Tuple[Sequence[Signal], Direction]],
        holdout_records: Sequence[Tuple[Sequence[Signal], Direction]],
        *,
        initial: Optional[Mapping[str, float]] = None,
    ) -> WeightCandidate:
        """Search weights on ``train_records``, then accept only if they also
        improve on ``holdout_records``.

        Accepting on the holdout split is what separates this from curve
        fitting: a weight vector that wins in-sample but loses out-of-sample is
        explicitly rejected and the incumbent is kept.
        """
        base = dict(initial or self.config.ensemble.weights)
        incumbent = evaluate_weights(train_records, base, config=self.config)
        best = dict(base)
        best_score = incumbent.score

        for _ in range(max(1, self.passes)):
            improved = False
            for name in WEIGHTED_PROVIDERS:
                current = best.get(name, 0.0)
                for delta in (self.step, -self.step):
                    candidate = dict(best)
                    candidate[name] = clamp(current + delta, self.min_weight, self.max_weight)
                    total = sum(candidate.values())
                    if total <= 0:
                        continue
                    scored = evaluate_weights(train_records, candidate, config=self.config)
                    if scored.score > best_score + 1e-9:
                        best, best_score = candidate, scored.score
                        improved = True
            if not improved:
                break

        incumbent_holdout = evaluate_weights(holdout_records, base, config=self.config)
        challenger_holdout = evaluate_weights(holdout_records, best, config=self.config)

        if challenger_holdout.score > incumbent_holdout.score + 1e-9:
            challenger_holdout.accepted = True
            challenger_holdout.reason = (
                f"holdout score {challenger_holdout.score:.4f} > incumbent "
                f"{incumbent_holdout.score:.4f}"
            )
            return challenger_holdout

        incumbent_holdout.accepted = False
        incumbent_holdout.reason = (
            f"challenger holdout score {challenger_holdout.score:.4f} did not beat "
            f"incumbent {incumbent_holdout.score:.4f}; keeping current weights"
        )
        return incumbent_holdout

    @staticmethod
    def grid(sizes: Sequence[float] = (0.10, 0.15, 0.20, 0.25)) -> List[Dict[str, float]]:
        """A small, symmetric grid of weight vectors for offline comparison.

        Deliberately tiny: the point is to show that the initialisation is
        reasonable, not to search a high-dimensional space.
        """
        out: List[Dict[str, float]] = []
        for combo in itertools.product(sizes, repeat=2):
            tech, pa = combo
            remaining = max(0.0, 1.0 - tech - pa)
            out.append({
                "technical": tech,
                "price_action": pa,
                "structure": remaining * 0.25,
                "regime": remaining * 0.25,
                "ml": remaining * 0.35,
                "vision": remaining * 0.15,
            })
        return out
