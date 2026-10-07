"""Outcome resolution: turning a logged prediction into a realised result.

This module is the ONLY place allowed to read candles after a prediction's
``as_of`` timestamp.  Keeping that privilege in one small, well-tested module
is what makes the leakage audit tractable.

A note on NO-TRADE outcomes
---------------------------
A NO-TRADE decision is not automatically "correct".  If price subsequently
moved decisively in one direction, the system *missed* a trade
(``NO_TRADE_MISSED``); if price chopped sideways, abstaining was genuinely
correct (``NO_TRADE_CORRECT``).  Recording this distinction is what lets the
experience layer learn whether the gate is too conservative or too loose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..types import (
    Candle,
    Decision,
    Direction,
    ErrorCategory,
    MarketSeries,
    Outcome,
    Prediction,
    Timeframe,
)
from ..utils.logging import get_logger
from ..utils.stats import safe_div

__all__ = ["classify_outcome", "OutcomeResolver", "DEFAULT_AMBIGUITY_THRESHOLD"]

_log = get_logger(__name__)

#: Relative move below which a NO-TRADE is considered genuinely correct.
#: 5 bps is a deliberately tight bar: a NO-TRADE only earns "correct" when
#: price really did nothing.  Anything larger counts as a missed move.
DEFAULT_AMBIGUITY_THRESHOLD = 0.0005


def classify_outcome(
    decision: Decision,
    actual: Direction,
    *,
    relative_move: float = 0.0,
    ambiguity_threshold: float = DEFAULT_AMBIGUITY_THRESHOLD,
) -> ErrorCategory:
    """Map a (decision, realised direction) pair onto an error category."""
    if decision is Decision.UP:
        return ErrorCategory.CORRECT_UP if actual is Direction.UP else ErrorCategory.FALSE_UP
    if decision is Decision.DOWN:
        return (
            ErrorCategory.CORRECT_DOWN if actual is Direction.DOWN else ErrorCategory.FALSE_DOWN
        )
    # NO-TRADE: correct only when the market genuinely did not commit.
    if actual is Direction.NEUTRAL or abs(relative_move) <= ambiguity_threshold:
        return ErrorCategory.NO_TRADE_CORRECT
    return ErrorCategory.NO_TRADE_MISSED


@dataclass(frozen=True)
class ResolutionWindow:
    """The candles a prediction is allowed to be settled against."""

    candles: Tuple[Candle, ...]

    @property
    def empty(self) -> bool:
        return not self.candles

    @property
    def exit_price(self) -> Optional[float]:
        return self.candles[-1].close if self.candles else None

    @property
    def high(self) -> Optional[float]:
        return max(c.high for c in self.candles) if self.candles else None

    @property
    def low(self) -> Optional[float]:
        return min(c.low for c in self.candles) if self.candles else None


class OutcomeResolver:
    """Settles predictions against the market data that followed them."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.payout = self.config.risk.payout
        self.ambiguity_threshold = DEFAULT_AMBIGUITY_THRESHOLD

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def window_for(
        prediction: Prediction, series: MarketSeries
    ) -> ResolutionWindow:
        """Candles strictly after the prediction's ``as_of``, up to expiry.

        A candle is usable only once it has *closed* at or before the expiry
        instant, mirroring exactly how the prediction path treats history.
        """
        usable = tuple(
            c
            for c in series.candles
            if c.timestamp >= prediction.timestamp + prediction.timeframe.seconds
            and c.close_time <= prediction.expires_at
        )
        return ResolutionWindow(usable)

    def resolve(
        self,
        prediction: Prediction,
        series: MarketSeries,
        *,
        payout: Optional[float] = None,
        stake: Optional[float] = None,
    ) -> Optional[Outcome]:
        """Resolve one prediction, or return ``None`` if it is not yet due.

        Returning ``None`` (rather than a fabricated result) is important:
        an unresolved prediction must never be silently treated as a loss.
        """
        window = self.window_for(prediction, series)
        if window.empty:
            return None
        exit_price = window.exit_price
        if exit_price is None:
            return None
        entry = float(prediction.entry_price)
        relative_move = safe_div(exit_price - entry, entry, 0.0)
        if relative_move > 0:
            actual = Direction.UP
        elif relative_move < 0:
            actual = Direction.DOWN
        else:
            actual = Direction.NEUTRAL

        category = classify_outcome(
            prediction.decision,
            actual,
            relative_move=relative_move,
            ambiguity_threshold=self.ambiguity_threshold,
        )

        applied_payout = self.payout if payout is None else float(payout)
        pnl = self._profit_or_loss(prediction.decision, category, applied_payout, stake)

        return Outcome(
            prediction_id=prediction.prediction_id,
            resolved_at=window.candles[-1].close_time,
            actual_direction=actual,
            exit_price=exit_price,
            error_category=category,
            profit_or_loss=pnl,
            payout=applied_payout,
            note=f"relative_move={relative_move:.6f}",
        )

    @staticmethod
    def _profit_or_loss(
        decision: Decision,
        category: ErrorCategory,
        payout: float,
        stake: Optional[float],
    ) -> float:
        """P/L in *units of stake* for directional calls, 0.0 for NO-TRADE.

        A NO-TRADE has no stake at risk, so its P/L is exactly zero - this is
        what stops abstention from being scored as a loss.
        """
        if decision is Decision.NO_TRADE:
            return 0.0
        base = 1.0 if stake is None else float(stake)
        if category in (ErrorCategory.CORRECT_UP, ErrorCategory.CORRECT_DOWN):
            return base * payout
        if category in (ErrorCategory.FALSE_UP, ErrorCategory.FALSE_DOWN):
            return -base
        return 0.0

    # -- batch -------------------------------------------------------------
    def resolve_pending(
        self,
        predictions: Sequence[Prediction],
        series_by_key: Mapping[Tuple[str, Timeframe], MarketSeries],
        *,
        payout: Optional[float] = None,
    ) -> List[Outcome]:
        """Resolve every prediction that has enough subsequent data.

        Predictions without a matching series, or whose horizon has not
        elapsed, are skipped rather than guessed at.
        """
        out: List[Outcome] = []
        for prediction in predictions:
            series = series_by_key.get((prediction.asset, prediction.timeframe))
            if series is None:
                _log.debug("no series for %s %s", prediction.asset, prediction.timeframe)
                continue
            outcome = self.resolve(prediction, series, payout=payout)
            if outcome is not None:
                out.append(outcome)
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "payout": self.payout,
            "ambiguity_threshold": self.ambiguity_threshold,
            "breakeven_win_rate": round(1.0 / (1.0 + self.payout), 4) if self.payout > 0 else None,
        }
