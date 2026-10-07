"""Tests for the pure-Python logistic regression and shared base helpers."""

import math
import random
import unittest

from market_ai.models.base import (
    impute,
    predict_labels,
    standardize_apply,
    standardize_fit,
    to_float,
    train_test_time_split,
)
from market_ai.models.linear import LogisticRegressionModel
from market_ai.utils.stats import expected_calibration_error


def _separable(n=400, d=3, seed=0):
    """Linearly separable synthetic data with both classes present."""
    rng = random.Random(seed)
    X = []
    y = []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.0) for _ in range(d)]
        score = 0.5 * row[0] + row[1] + row[2 % d]
        X.append(row)
        y.append(1 if score > 0.0 else 0)
    return X, y


def _noisy(n=1500, seed=1):
    """Data drawn from a true logistic model (well-calibrated by design)."""
    rng = random.Random(seed)
    X = []
    y = []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.0) for _ in range(3)]
        z = 1.0 * row[0] + 0.6 * row[1]
        p = 1.0 / (1.0 + math.exp(-z))
        X.append(row)
        y.append(1 if rng.random() < p else 0)
    return X, y


class BaseHelperTests(unittest.TestCase):
    def test_impute_median_fills_nan(self):
        out = impute([[1.0, float("nan")], [3.0, 4.0]], strategy="median")
        self.assertEqual(out[0][1], 4.0)

    def test_impute_mean_and_zero(self):
        self.assertEqual(impute([[0.0, float("nan")], [2.0, 5.0]], "mean")[0][1], 5.0)
        self.assertEqual(impute([[0.0, float("nan")]], "zero")[0][1], 0.0)

    def test_impute_rejects_ragged(self):
        with self.assertRaises(ValueError):
            impute([[1.0, 2.0], [1.0]])

    def test_standardize_centres_columns(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        means, stds = standardize_fit(X)
        Z = standardize_apply(X, means, stds)
        for col in range(2):
            self.assertAlmostEqual(sum(r[col] for r in Z) / 3.0, 0.0, places=9)

    def test_standardize_zero_variance_column(self):
        _, stds = standardize_fit([[1.0], [1.0], [1.0]])
        self.assertEqual(stds[0], 1.0)

    def test_time_split_is_ordered_and_disjoint(self):
        X = [[i] for i in range(10)]
        y = list(range(10))
        Xtr, ytr, Xte, yte = train_test_time_split(X, y, 0.6)
        self.assertEqual(len(Xtr), 6)
        self.assertEqual(ytr, [0, 1, 2, 3, 4, 5])
        self.assertEqual(yte, [6, 7, 8, 9])
        self.assertEqual(Xtr[-1][0], 5)
        self.assertEqual(Xte[0][0], 6)

    def test_time_split_bad_frac(self):
        with self.assertRaises(ValueError):
            train_test_time_split([[1.0]], [0], 1.0)

    def test_predict_labels_threshold(self):
        self.assertEqual(predict_labels([0.4, 0.5, 0.9]), [0, 1, 1])

    def test_to_float(self):
        self.assertTrue(math.isnan(to_float("abc")))
        self.assertEqual(to_float(2), 2.0)


class LogisticRegressionTests(unittest.TestCase):
    def test_converges_on_separable(self):
        X, y = _separable()
        model = LogisticRegressionModel().fit(X, y)
        probs = model.predict_proba(X)
        acc = sum(1 for p, t in zip(probs, y) if (p >= 0.5) == (t == 1)) / len(y)
        self.assertGreater(acc, 0.9)

    def test_probabilities_in_unit_interval(self):
        X, y = _separable()
        probs = LogisticRegressionModel().fit(X, y).predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_determinism_same_seed(self):
        X, y = _separable()
        a = LogisticRegressionModel(seed=5).fit(X, y).predict_proba(X)
        b = LogisticRegressionModel(seed=5).fit(X, y).predict_proba(X)
        self.assertEqual(a, b)

    def test_feature_importance(self):
        X, y = _separable()
        model = LogisticRegressionModel().fit(X, y, feature_names=["a", "b", "c"])
        imp = model.feature_importance()
        self.assertEqual(set(imp), {"a", "b", "c"})
        self.assertGreater(sum(imp.values()), 0.0)

    def test_get_set_params(self):
        model = LogisticRegressionModel(epochs=10)
        self.assertEqual(model.get_params()["epochs"], 10)
        model.set_params(epochs=20)
        self.assertEqual(model.epochs, 20)
        with self.assertRaises(ValueError):
            model.set_params(bogus=1)

    def test_predict_labels_method(self):
        X, y = _separable()
        model = LogisticRegressionModel().fit(X, y)
        self.assertEqual(set(model.predict(X)), {0, 1})

    def test_nan_features_handled(self):
        X, y = _separable()
        Xn = [list(r) for r in X]
        for i in range(0, len(Xn), 5):
            Xn[i][0] = float("nan")
        model = LogisticRegressionModel().fit(Xn, y)
        probs = model.predict_proba(Xn)
        self.assertTrue(all(math.isfinite(p) for p in probs))

    def test_single_class_target(self):
        X, _ = _separable()
        zeros = LogisticRegressionModel().fit(X, [0] * len(X))
        self.assertTrue(all(p == 0.0 for p in zeros.predict_proba(X)))
        ones = LogisticRegressionModel().fit(X, [1] * len(X))
        self.assertTrue(all(p == 1.0 for p in ones.predict_proba(X)))

    def test_calibrated_on_noisy_data(self):
        X, y = _noisy()
        probs = LogisticRegressionModel().fit(X, y).predict_proba(X)
        self.assertLess(expected_calibration_error(probs, y, 10), 0.15)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            LogisticRegressionModel().predict_proba([[1.0]])


if __name__ == "__main__":
    unittest.main()

