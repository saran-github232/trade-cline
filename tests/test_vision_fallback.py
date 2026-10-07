"""Tests for ChartVisionAnalyser provider selection and fallback behaviour.

These tests never touch the network: external providers either fail fast on
missing credentials or are replaced with a registered fake.  ``urlopen`` is
patched to blow up if anything tries to make a real request.
"""

import os
import tempfile
import unittest
from unittest import mock

from market_ai.config import Config, VisionConfig
from market_ai.types import Candle, MarketSeries, Timeframe
from market_ai.vision.analyser import ChartVisionAnalyser
from market_ai.vision.providers import VisionProviderError, register_provider


def make_series(closes):
    """Build a small synthetic MarketSeries."""
    candles = []
    start = 1_700_000_000
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i > 0 else close
        high = max(open_, close) + 0.2
        low = min(open_, close) - 0.2
        candles.append(
            Candle(
                timestamp=start + i * 60,
                asset="TEST",
                timeframe=Timeframe.M1,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=1.0,
            )
        )
    return MarketSeries(asset="TEST", timeframe=Timeframe.M1, candles=tuple(candles))


def _uptrend(n=120):
    return [100.0 + 0.5 * i for i in range(n)]


class _AlwaysFailProvider:
    """A provider that always raises, standing in for a broken backend."""

    name = "alwaysfail"

    def analyse(self, *, image_path=None, summary=None, timeframe=None, asset=None):
        raise VisionProviderError("simulated provider failure")


def _config_with_provider(name):
    return Config().with_overrides(vision=VisionConfig(provider=name))


class VisionFallbackTests(unittest.TestCase):
    def setUp(self):
        self.series = make_series(_uptrend())
        register_provider("alwaysfail", lambda cfg: _AlwaysFailProvider())
        self._env = mock.patch.dict(
            os.environ,
            {
                "MARKET_AI_VISION_API_KEY": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_offline_provider_is_default_and_not_degraded(self):
        analyser = ChartVisionAnalyser(Config())
        result = analyser.analyse(series=self.series, summary={})
        self.assertEqual(result.provider, "offline")
        self.assertFalse(result.degraded)

    def test_failing_provider_falls_back_to_offline_degraded(self):
        analyser = ChartVisionAnalyser(_config_with_provider("alwaysfail"))
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "chart.png")
            with open(image, "wb") as handle:
                handle.write(b"\x89PNG\r\n")
            with mock.patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("no network allowed in tests"),
            ):
                result = analyser.analyse(
                    series=self.series, summary={}, image_path=image
                )
        self.assertTrue(result.degraded)
        self.assertEqual(result.provider, "offline")
        self.assertIn("provider error", analyser.last_fallback_reason)

    def test_missing_image_path_degrades_without_raising(self):
        analyser = ChartVisionAnalyser(_config_with_provider("openai"))
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("no network allowed in tests"),
        ):
            result = analyser.analyse(
                series=self.series,
                summary={},
                image_path="/nonexistent/path/to/chart.png",
            )
        self.assertTrue(result.degraded)
        self.assertEqual(result.provider, "offline")

    def test_unsupported_extension_degrades(self):
        analyser = ChartVisionAnalyser(_config_with_provider("openai"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "chart.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("not an image")
            result = analyser.analyse(series=self.series, summary={}, image_path=path)
        self.assertTrue(result.degraded)

    def test_corrupt_image_path_degrades_without_raising(self):
        analyser = ChartVisionAnalyser(_config_with_provider("openai"))
        with tempfile.TemporaryDirectory() as tmp:
            # A directory is a valid existing path but not a usable image file.
            result = analyser.analyse(
                series=self.series, summary={}, image_path=tmp
            )
        self.assertTrue(result.degraded)
        self.assertEqual(result.provider, "offline")

    def test_no_credentials_degrades_without_network(self):
        analyser = ChartVisionAnalyser(_config_with_provider("openai"))
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "chart.png")
            with open(image, "wb") as handle:
                handle.write(b"fake-png-bytes")
            with mock.patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("no network allowed in tests"),
            ):
                result = analyser.analyse(
                    series=self.series, summary={}, image_path=image
                )
        self.assertTrue(result.degraded)
        self.assertEqual(result.provider, "offline")

    def test_raw_response_is_stored(self):
        analyser = ChartVisionAnalyser(Config())
        result = analyser.analyse(series=self.series, summary={})
        self.assertEqual(analyser.last_result.to_dict(), result.to_dict())
        self.assertEqual(analyser.last_raw_response, dict(result.raw))


if __name__ == "__main__":
    unittest.main()
