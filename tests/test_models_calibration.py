"""Tests for probability calibration and reliability reporting."""

import math
import random
import unittest

from market_ai.models.base import sigmoid
from market_ai.models.calibration import Calibrator, calibration_report


def _miscalibrated(n=4000, seed=1):
    """Overconfident scores: true logit is scaled by 2.0 before sigmoid."""
    rng = random.Random(seed)
    probs = []
    y = []
    for _ in range(n):
        z = rng.gauss(0.0, 1.2)
        true_p = sigmoid(z)
        outcome = 1 if rng.random() < true_p else 0
        probs.append(sigmoid(2.0 * z))  # overconfident
        y.append(outcome)
    return probs, y


class CalibrationReportTests(unittest.TestCase):
    def test_report_keys_and_bins(self):
        probs, y = _miscalibrated(n=1000)
        report = calibration_report(probs, y, n_bins=10)
        self.assertEqual(len(report["bins"]), 10)
        self.assertEqual(sum(b["count"] for b in report["bins"]), 1000)
        self.assertTrue(math.isfinite(report["ece"]))
        self.assertTrue(math.isfinite(report["brier"]))
        self.assertTrue(math.isfinite(report["log_loss"]))
        self.assertEqual(report["n"], 1000)

    def test_report_length_mismatch(self):
        with self.assertRaises(ValueError):
            calibration_report([0.1, 0.2], [1])


class CalibratorTests(unittest.TestCase):
    def test_isotonic_reduces_ece(self):
        probs, y = _miscalibrated()
        before = calibration_report(probs, y)["ece"]
        cal = Calibrator(method="isotonic").fit(probs, y)
        after = calibration_report(cal.transform(probs), y)["ece"]
        self.assertLess(after, before)

    def test_platt_reduces_ece(self):
        probs, y = _miscalibrated()
        before = calibration_report(probs, y)["ece"]
        cal = Calibrator(method="platt").fit(probs, y)
        after = calibration_report(cal.transform(probs), y)["ece"]
        self.assertLess(after, before)

    def test_transform_in_unit_interval(self):
        probs, y = _miscalibrated(n=500)
        for method in ("platt", "isotonic"):
            cal = Calibrator(method=method).fit(probs, y)
            out = cal.transform([0.0, 0.25, 0.5, 0.75, 1.0])
            self.assertTrue(all(0.0 <= p <= 1.0 for p in out))

    def test_isotonic_is_monotone(self):
        probs, y = _miscalibrated(n=2000)
        cal = Calibrator(method="isotonic").fit(probs, y)
        grid = [i / 100.0 for i in range(101)]
        out = cal.transform(grid)
        for a, b in zip(out, out[1:]):
            self.assertLessEqual(a, b + 1e-9)

    def test_roundtrip_platt(self):
        probs, y = _miscalibrated(n=500)
        cal = Calibrator(method="platt").fit(probs, y)
        restored = Calibrator.from_dict(cal.to_dict())
        self.assertEqual(cal.transform(probs), restored.transform(probs))

    def test_roundtrip_isotonic(self):
        probs, y = _miscalibrated(n=500)
        cal = Calibrator(method="isotonic").fit(probs, y)
        restored = Calibrator.from_dict(cal.to_dict())
        self.assertEqual(cal.transform(probs), restored.transform(probs))

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            Calibrator(method="magic")

    def test_transform_before_fit_raises(self):
        with self.assertRaises(ValueError):
            Calibrator().transform([0.5])


if __name__ == "__main__":
    unittest.main()
