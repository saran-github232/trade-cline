"""Tests for the deterministic offline chart-vision analyser and renderer."""

import math
import os
import tempfile
import unittest

from market_ai.types import Candle, Direction, MarketSeries, Timeframe
from market_ai.vision.chart_render import render_chart_svg
from market_ai.vision.offline import OfflineChartAnalyser


def make_series(closes, *, asset="TEST", timeframe=Timeframe.M1):
    """Build a synthetic MarketSeries with valid OHLC geometry."""
    candles = []
    start = 1_700_000_000
    step = timeframe.seconds
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i > 0 else close
        body = abs(close - open_)
        high = max(open_, close) + body * 0.5 + 0.1
        low = min(open_, close) - body * 0.5 - 0.1
        candles.append(
            Candle(
                timestamp=start + i * step,
                asset=asset,
                timeframe=timeframe,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=100.0,
            )
        )
    return MarketSeries(asset=asset, timeframe=timeframe, candles=tuple(candles))


def uptrend(n=120):
    return [100.0 + 0.5 * i + 0.8 * math.sin(i / 5.0) for i in range(n)]


def downtrend(n=120):
    return [100.0 - 0.5 * i + 0.8 * math.sin(i / 5.0) for i in range(n)]


def ranging(n=120):
    return [100.0 + 0.2 * math.sin(i / 6.0) for i in range(n)]


class OfflineAnalyserTests(unittest.TestCase):
    def setUp(self):
        self.analyser = OfflineChartAnalyser()

    def _analyse(self, closes):
        series = make_series(closes)
        return series, self.analyser.analyse(series=series, summary={})

    def test_uptrend_is_bullish(self):
        _, result = self._analyse(uptrend())
        self.assertEqual(result.direction_bias, Direction.UP)
        self.assertEqual(result.trend, "up")
        self.assertEqual(result.provider, "offline")
        self.assertFalse(result.degraded)

    def test_downtrend_is_bearish(self):
        _, result = self._analyse(downtrend())
        self.assertEqual(result.direction_bias, Direction.DOWN)
        self.assertEqual(result.trend, "down")

    def test_ranging_is_neutral(self):
        _, result = self._analyse(ranging())
        self.assertEqual(result.direction_bias, Direction.NEUTRAL)

    def test_confidence_in_unit_interval(self):
        for closes in (uptrend(), downtrend(), ranging()):
            _, result = self._analyse(closes)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_levels_are_within_observed_range(self):
        for closes in (uptrend(), downtrend(), ranging()):
            series, result = self._analyse(closes)
            low = min(series.lows)
            high = max(series.highs)
            for level in list(result.support) + list(result.resistance):
                self.assertGreaterEqual(level, low - 1e-9)
                self.assertLessEqual(level, high + 1e-9)

    def test_probabilities_in_unit_interval(self):
        _, result = self._analyse(uptrend())
        self.assertGreaterEqual(result.breakout_probability, 0.0)
        self.assertLessEqual(result.breakout_probability, 1.0)
        self.assertGreaterEqual(result.reversal_probability, 0.0)
        self.assertLessEqual(result.reversal_probability, 1.0)

    def test_invalidating_conditions_present(self):
        _, result = self._analyse(uptrend())
        self.assertTrue(result.invalidating_conditions)
        self.assertTrue(result.reasoning_summary)

    def test_result_is_well_formed(self):
        _, result = self._analyse(uptrend())
        as_dict = result.to_dict()
        for key in (
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
        ):
            self.assertIn(key, as_dict)

    def test_degenerate_series_does_not_raise(self):
        series = make_series([100.0])
        result = self.analyser.analyse(series=series, summary={})
        self.assertEqual(result.direction_bias, Direction.NEUTRAL)
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.degraded)

    def test_empty_series_does_not_raise(self):
        series = MarketSeries(asset="TEST", timeframe=Timeframe.M1, candles=())
        result = self.analyser.analyse(series=series, summary={})
        self.assertEqual(result.direction_bias, Direction.NEUTRAL)


class ChartRenderTests(unittest.TestCase):
    def test_render_writes_svg(self):
        series = make_series(uptrend(20))
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "chart.svg")
            written = render_chart_svg(series, out, title="Test")
            self.assertEqual(written, out)
            with open(out, "r", encoding="utf-8") as handle:
                content = handle.read()
        self.assertIn("<svg", content)
        self.assertIn("Test", content)

    def test_render_empty_series_is_valid(self):
        series = MarketSeries(asset="TEST", timeframe=Timeframe.M1, candles=())
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "empty.svg")
            render_chart_svg(series, out)
            with open(out, "r", encoding="utf-8") as handle:
                content = handle.read()
        self.assertIn("No data", content)

    def test_render_is_deterministic(self):
        series = make_series(downtrend(15))
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.svg")
            b = os.path.join(tmp, "b.svg")
            render_chart_svg(series, a)
            render_chart_svg(series, b)
            self.assertEqual(
                open(a, encoding="utf-8").read(), open(b, encoding="utf-8").read()
            )



if __name__ == "__main__":
    unittest.main()
