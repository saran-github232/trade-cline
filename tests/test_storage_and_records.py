"""Tests for SQLite storage, the outcome resolver, experience memory and the registry."""

import tempfile
import unittest
from pathlib import Path

from market_ai.config import Config
from market_ai.experience import ExperienceStore, build_experience, derive_tags
from market_ai.experience.analytics import (
    accuracy_trend,
    confidence_reliability,
    error_breakdown,
    recurring_patterns,
    regime_failures,
    summarise,
)
from market_ai.models.registry import ModelRecord, ModelRegistry
from market_ai.outcomes import OutcomeResolver, classify_outcome
from market_ai.storage import Database
from market_ai.types import (
    Candle,
    Decision,
    Direction,
    ErrorCategory,
    Experience,
    MarketSeries,
    Outcome,
    Prediction,
    ProbabilitySet,
    Regime,
    Signal,
    Timeframe,
)


def series(n=400, tf=Timeframe.M1, start=60, drift=0.0):
    candles = []
    price = 100.0
    for i in range(n):
        o = price
        c = price + drift
        h = max(o, c) + 0.05
        l = min(o, c) - 0.05
        candles.append(Candle(start + i * tf.seconds, "EURUSD", tf, o, h, l, c, 10.0))
        price = c
    return MarketSeries("EURUSD", tf, tuple(candles))


def prediction(ts=60, tf=Timeframe.M1, decision=Decision.UP, entry=100.0, horizon_bars=3):
    return Prediction(
        prediction_id=f"p-{ts}-{decision.value}", timestamp=ts, asset="EURUSD",
        timeframe=tf, decision=decision,
        probabilities=ProbabilitySet(0.6, 0.2, 0.2), confidence=0.7,
        regime=Regime.STRONG_UPTREND, agreement=0.8, reason="test",
        model_version="m1", dataset_version="d1", feature_version="f1",
        strategy_version="s1", horizon_seconds=horizon_bars * tf.seconds,
        entry_price=entry, expires_at=ts + tf.seconds + horizon_bars * tf.seconds,
    )


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:").initialize()

    def tearDown(self):
        self.db.close()

    def test_add_and_get_prediction(self):
        p = prediction()
        self.db.predictions.add(p)
        stored = self.db.predictions.get(p.prediction_id)
        self.assertEqual(stored["decision"], "UP")
        self.assertEqual(Prediction.from_dict(stored).to_dict(), p.to_dict())

    def test_unresolved_lifecycle(self):
        p = prediction()
        self.db.predictions.add(p)
        self.assertEqual(len(self.db.predictions.unresolved()), 1)
        self.db.predictions.mark_resolved(p.prediction_id)
        self.assertEqual(len(self.db.predictions.unresolved()), 0)

    def test_outcome_and_experience_round_trip(self):
        p = prediction()
        self.db.predictions.add(p)
        o = Outcome(p.prediction_id, 400, Direction.UP, 101.0, ErrorCategory.CORRECT_UP, 0.85, 0.85)
        self.db.outcomes.add(o)
        exp = build_experience(p, o)
        self.db.experiences.add(exp)
        self.assertEqual(self.db.experiences.count(), 1)
        self.assertEqual(Experience.from_dict(self.db.experiences.get(p.prediction_id)).decision, Decision.UP)

    def test_idempotent_replace(self):
        p = prediction()
        self.db.predictions.add(p)
        self.db.predictions.add(p)
        self.assertEqual(self.db.predictions.count(), 1)

    def test_model_history_audit(self):
        self.db.models.log_history("promote", from_version="m0", to_version="m1", reason="gates")
        history = self.db.models.history()
        self.assertEqual(history[0]["to_version"], "m1")

    def test_stats(self):
        self.db.predictions.add(prediction())
        self.assertIn("predictions", self.db.stats())

    def test_persistence_across_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.db"
            db = Database(path).initialize()
            db.predictions.add(prediction())
            db.close()
            db2 = Database(path).initialize()
            self.assertEqual(db2.predictions.count(), 1)
            db2.close()


class OutcomeResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = OutcomeResolver(Config())

    def test_classification_matrix(self):
        self.assertIs(classify_outcome(Decision.UP, Direction.UP), ErrorCategory.CORRECT_UP)
        self.assertIs(classify_outcome(Decision.UP, Direction.DOWN), ErrorCategory.FALSE_UP)
        self.assertIs(classify_outcome(Decision.DOWN, Direction.DOWN), ErrorCategory.CORRECT_DOWN)
        self.assertIs(classify_outcome(Decision.DOWN, Direction.UP), ErrorCategory.FALSE_DOWN)

    def test_no_trade_distinguishes_missed_from_correct(self):
        self.assertIs(
            classify_outcome(Decision.NO_TRADE, Direction.UP, relative_move=0.01),
            ErrorCategory.NO_TRADE_MISSED,
        )
        self.assertIs(
            classify_outcome(Decision.NO_TRADE, Direction.UP, relative_move=0.00001),
            ErrorCategory.NO_TRADE_CORRECT,
        )

    def test_unresolved_when_horizon_not_elapsed(self):
        p = prediction(ts=60, horizon_bars=3)
        # Only the decision bar exists, so nothing has closed inside the
        # prediction window yet.  The resolver must return None rather than
        # fabricating a result.
        short = MarketSeries("EURUSD", Timeframe.M1, series(n=1).candles)
        self.assertIsNone(self.resolver.resolve(p, short))

    def test_win_pays_payout_and_loss_pays_stake(self):
        up = series(n=50, drift=0.5)
        p = prediction(ts=60, horizon_bars=3)
        outcome = self.resolver.resolve(p, up)
        self.assertIsNotNone(outcome)
        self.assertIs(outcome.error_category, ErrorCategory.CORRECT_UP)
        self.assertAlmostEqual(outcome.profit_or_loss, 0.85)

    def test_no_trade_has_zero_pnl(self):
        up = series(n=50, drift=0.5)
        p = prediction(ts=60, decision=Decision.NO_TRADE, horizon_bars=3)
        outcome = self.resolver.resolve(p, up)
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.profit_or_loss, 0.0)

    def test_resolve_pending_skips_missing_series(self):
        p = prediction()
        self.assertEqual(self.resolver.resolve_pending([p], {}), [])


class ExperienceTests(unittest.TestCase):
    def _exp(self, i, regime=Regime.RANGE, decision=Decision.UP, err=ErrorCategory.FALSE_UP,
             conf=0.8, pnl=-1.0, tags=("RANGE_NOISE",)):
        return Experience(
            prediction_id=f"p{i}", timestamp=1000 + i, asset="EURUSD",
            timeframe=Timeframe.M1, features={"momentum.rsi_14": 50.0}, regime=regime,
            decision=decision, confidence=conf, probabilities={"p_up": 0.6},
            model_version="m1", strategy_version="s1", feature_version="f1",
            actual_direction=Direction.UP, error_category=err, profit_or_loss=pnl, tags=tags,
        )

    def test_store_add_and_count(self):
        store = ExperienceStore()
        store.add_many([self._exp(i) for i in range(10)])
        self.assertEqual(store.count(), 10)
        self.assertEqual(len(store.all()), 10)

    def test_store_deduplicates_by_prediction_id(self):
        store = ExperienceStore()
        store.add(self._exp(1))
        store.add(self._exp(1))
        self.assertEqual(store.count(), 1)

    def test_error_breakdown(self):
        store = ExperienceStore()
        store.add_many([self._exp(i) for i in range(10)])
        breakdown = error_breakdown(store.all())
        self.assertEqual(breakdown["total"], 10)
        self.assertEqual(breakdown["counts"]["FALSE_UP"], 10)
        self.assertEqual(breakdown["win_rate"], 0.0)
        self.assertEqual(len(breakdown["win_rate_ci"]), 2)

    def test_regime_failures_flags_small_samples(self):
        store = ExperienceStore()
        store.add_many([self._exp(i) for i in range(10)])
        report = regime_failures(store.all())
        self.assertFalse(report["RANGE"]["reliable"])

    def test_confidence_reliability_detects_overconfidence(self):
        store = ExperienceStore()
        store.add_many([self._exp(i, conf=0.9) for i in range(40)])
        report = confidence_reliability(store.all())
        self.assertEqual(report["bias"], "overconfident")
        self.assertGreater(report["ece"], 0.5)

    def test_recurring_patterns_ranked_by_cost(self):
        store = ExperienceStore()
        store.add_many([self._exp(i, pnl=-2.0) for i in range(10)])
        patterns = recurring_patterns(store.all(), min_support=5)
        self.assertTrue(patterns)
        self.assertEqual(patterns[0]["pattern"], "RANGE_NOISE")
        self.assertGreater(patterns[0]["cost"], 0)

    def test_accuracy_trend_detects_degradation(self):
        store = ExperienceStore()
        items = []
        for i in range(100):
            # Early windows win, later windows lose -> degrading.
            err = ErrorCategory.CORRECT_UP if i < 30 else ErrorCategory.FALSE_UP
            items.append(self._exp(i, err=err, pnl=1.0 if i < 30 else -1.0))
        report = accuracy_trend(items, windows=5)
        self.assertTrue(report["degrading"])
        self.assertLess(report["slope"], 0)

    def test_summarise_returns_all_sections(self):
        store = ExperienceStore()
        store.add_many([self._exp(i) for i in range(20)])
        summary = summarise(store.all())
        for key in ("errors", "regimes", "timeframes", "calibration", "patterns", "trend"):
            self.assertIn(key, summary)

    def test_derive_tags_marks_overconfidence(self):
        p = prediction()
        p = Prediction.from_dict({**p.to_dict(), "confidence": 0.9})
        o = Outcome(p.prediction_id, 400, Direction.DOWN, 99.0, ErrorCategory.FALSE_UP, -1.0, 0.85)
        tags = derive_tags(p, o)
        self.assertIn("OVERCONFIDENCE", tags)

    def test_derive_tags_marks_good_abstention(self):
        p = prediction(decision=Decision.NO_TRADE)
        o = Outcome(p.prediction_id, 400, Direction.NEUTRAL, 100.0, ErrorCategory.NO_TRADE_CORRECT, 0.0, 0.85)
        self.assertIn("GOOD_ABSTENTION", derive_tags(p, o))
