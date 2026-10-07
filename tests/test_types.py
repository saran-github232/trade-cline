"""Tests for the core domain types."""

import math
import unittest

from market_ai.types import (
    Candle,
    Decision,
    Direction,
    ErrorCategory,
    FeatureVector,
    MarketSeries,
    Outcome,
    Prediction,
    ProbabilitySet,
    Regime,
    Signal,
    Timeframe,
    epoch_to_iso,
    iso_to_epoch,
    to_utc_epoch,
)


def make_candle(ts=60, o=1.0, h=1.2, l=0.9, c=1.1, tf=Timeframe.M1):
    return Candle(ts, "EURUSD", tf, o, h, l, c, 100.0)


class TimeframeTests(unittest.TestCase):
    def test_seconds_and_rank(self):
        self.assertEqual(Timeframe.M1.seconds, 60)
        self.assertEqual(Timeframe.H1.seconds, 3600)
        self.assertLess(Timeframe.M1.rank, Timeframe.H1.rank)

    def test_parse_aliases(self):
        for text in ("1m", "M1", "60s", "1min"):
            self.assertIs(Timeframe.parse(text), Timeframe.M1)
        self.assertIs(Timeframe.parse("1H"), Timeframe.H1)

    def test_parse_rejects_unknown(self):
        with self.assertRaises(ValueError):
            Timeframe.parse("3m")


class TimestampTests(unittest.TestCase):
    def test_epoch_passthrough(self):
        self.assertEqual(to_utc_epoch(1700000000), 1700000000)
        self.assertEqual(to_utc_epoch(1700000000.7), 1700000000)

    def test_iso_round_trip(self):
        epoch = iso_to_epoch("2024-01-01T00:00:00Z")
        self.assertEqual(epoch_to_iso(epoch), "2024-01-01T00:00:00Z")

    def test_naive_string_assumed_utc(self):
        # A naive timestamp must be read as UTC, never as local time.
        self.assertEqual(iso_to_epoch("2024-01-01T00:00:00"), iso_to_epoch("2024-01-01T00:00:00Z"))

    def test_offset_string_normalised(self):
        # 05:30+05:30 == 00:00Z
        self.assertEqual(iso_to_epoch("2024-01-01T05:30:00+05:30"), iso_to_epoch("2024-01-01T00:00:00Z"))

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            to_utc_epoch(True)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            to_utc_epoch(float("nan"))


class CandleTests(unittest.TestCase):
    def test_geometry(self):
        c = make_candle(o=1.0, h=1.3, l=0.8, c=1.2)
        self.assertAlmostEqual(c.body, 0.2)
        self.assertAlmostEqual(c.upper_wick, 0.1)
        self.assertAlmostEqual(c.lower_wick, 0.2)
        self.assertAlmostEqual(c.range, 0.5)
        self.assertTrue(c.is_bullish)
        self.assertFalse(c.is_bearish)
        self.assertTrue(c.is_valid())

    def test_close_time(self):
        self.assertEqual(make_candle(ts=60).close_time, 120)

    def test_invalid_ohlc_detected(self):
        bad = Candle(60, "X", Timeframe.M1, 1.0, 0.9, 1.1, 1.0)  # high < low
        self.assertFalse(bad.is_valid())

    def test_non_finite_rejected(self):
        with self.assertRaises(ValueError):
            Candle(60, "X", Timeframe.M1, 1.0, float("nan"), 1.0, 1.0)

    def test_round_trip(self):
        c = make_candle()
        self.assertEqual(Candle.from_dict(c.to_dict()), c)


class MarketSeriesTests(unittest.TestCase):
    def setUp(self):
        self.candles = [make_candle(ts=60 * i) for i in range(1, 11)]
        self.series = MarketSeries("EURUSD", Timeframe.M1, tuple(self.candles))

    def test_sorted(self):
        self.assertTrue(self.series.is_sorted())

    def test_visible_at_excludes_incomplete_bar(self):
        # Candle at ts=60 closes at 120.  As of 119 it must NOT be visible.
        visible = self.series.visible_at(119)
        self.assertNotIn(60, [c.timestamp for c in visible])
        visible = self.series.visible_at(120)
        self.assertIn(60, [c.timestamp for c in visible])

    def test_visible_at_never_returns_future(self):
        for as_of in range(0, 700, 17):
            visible = self.series.visible_at(as_of)
            self.assertTrue(all(c.close_time <= as_of for c in visible))

    def test_immutability_of_slice(self):
        original = self.series.candles
        sliced = self.series.slice(0, 3)
        self.assertEqual(self.series.candles, original)
        self.assertEqual(len(sliced), 3)


class SignalTests(unittest.TestCase):
    def test_clamping(self):
        s = Signal("x", Direction.UP, 1.5, -0.2, 0)
        self.assertEqual(s.probability, 1.0)
        self.assertEqual(s.confidence, 0.0)

    def test_direction_coercion(self):
        s = Signal("x", "DOWN", 0.2, 0.5, 0)
        self.assertIs(s.direction, Direction.DOWN)

    def test_round_trip(self):
        s = Signal("x", Direction.UP, 0.7, 0.4, 10, {"a": 1})
        self.assertEqual(Signal.from_dict(s.to_dict()).to_dict(), s.to_dict())


class ProbabilitySetTests(unittest.TestCase):
    def test_normalisation(self):
        p = ProbabilitySet(2.0, 1.0, 1.0)
        self.assertAlmostEqual(p.p_up + p.p_down + p.p_no_trade, 1.0)

    def test_best_prefers_no_trade_on_tie(self):
        self.assertIs(ProbabilitySet(0.4, 0.4, 0.4).best(), Decision.NO_TRADE)

    def test_best_directional(self):
        self.assertIs(ProbabilitySet(0.6, 0.2, 0.2).best(), Decision.UP)
        self.assertIs(ProbabilitySet(0.2, 0.6, 0.2).best(), Decision.DOWN)

    def test_invalid_total_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilitySet(0.0, 0.0, 0.0)


class RecordRoundTripTests(unittest.TestCase):
    def test_prediction_round_trip(self):
        p = Prediction(
            prediction_id="p1", timestamp=60, asset="EURUSD", timeframe=Timeframe.M1,
            decision=Decision.NO_TRADE, probabilities=ProbabilitySet(0.2, 0.2, 0.6),
            confidence=0.4, regime=Regime.RANGE, agreement=0.5, reason="weak",
            model_version="m1", dataset_version="d1", feature_version="f1",
            strategy_version="s1", horizon_seconds=180, entry_price=1.1, expires_at=360,
            signals=(Signal("technical", Direction.UP, 0.6, 0.5, 60),),
            features={"trend.ema_9": 1.05},
        )
        self.assertEqual(Prediction.from_dict(p.to_dict()).to_dict(), p.to_dict())

    def test_outcome_round_trip(self):
        o = Outcome("p1", 360, Direction.UP, 1.2, ErrorCategory.CORRECT_UP, 0.85, 0.85)
        self.assertEqual(Outcome.from_dict(o.to_dict()).to_dict(), o.to_dict())
        self.assertTrue(o.is_win)

    def test_feature_vector_nan_handling(self):
        fv = FeatureVector(60, "EURUSD", Timeframe.M1, "v", {"a": 1.0, "b": float("inf"), "c": "x"})
        self.assertEqual(fv.get("a"), 1.0)
        self.assertTrue(math.isnan(fv.get("b")))
        self.assertNotIn("c", fv.values)


class EnumTests(unittest.TestCase):
    def test_regime_trend_flag(self):
        self.assertTrue(Regime.STRONG_UPTREND.is_trend)
        self.assertFalse(Regime.RANGE.is_trend)

    def test_error_category_correctness(self):
        self.assertTrue(ErrorCategory.CORRECT_UP.is_correct)
        self.assertTrue(ErrorCategory.NO_TRADE_CORRECT.is_correct)
        self.assertFalse(ErrorCategory.FALSE_UP.is_correct)
        self.assertFalse(ErrorCategory.NO_TRADE_MISSED.is_correct)

    def test_direction_sign(self):
        self.assertEqual(Direction.UP.sign, 1)
        self.assertEqual(Direction.DOWN.sign, -1)
        self.assertEqual(Direction.NEUTRAL.sign, 0)


if __name__ == "__main__":
    unittest.main()
