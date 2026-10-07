"""Tests for the pure-Python gradient boosting model."""

import math
import random
import unittest

from market_ai.models.boosting import GradientBoostingModel


def _separable(n=400, d=3, seed=0):
    """Synthetic separable data with both classes represented."""
    rng = random.Random(seed)
    X = []
    y = []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.0) for _ in range(d)]
        score = row[0] + 0.5 * row[1] - 0.3 * row[2]
        X.append(row)
        y.append(1 if score > 0.0 else 0)
    return X, y


class GradientBoostingTests(unittest.TestCase):
    def test_learns_separable_data(self):
        X, y = _separable()
        model = GradientBoostingModel(n_estimators=60, seed=1).fit(X, y)
        probs = model.predict_proba(X)
        acc = sum(1 for p, t in zip(probs, y) if (p >= 0.5) == (t == 1)) / len(y)
        self.assertGreater(acc, 0.9)

    def test_probabilities_in_unit_interval(self):
        X, y = _separable()
        probs = GradientBoostingModel(n_estimators=30, seed=1).fit(X, y).predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_determinism_same_seed(self):
        X, y = _separable()
        a = GradientBoostingModel(n_estimators=30, seed=4).fit(X, y).predict_proba(X)
        b = GradientBoostingModel(n_estimators=30, seed=4).fit(X, y).predict_proba(X)
        self.assertEqual(a, b)

    def test_feature_importance_sums_positive(self):
        X, y = _separable()
        model = GradientBoostingModel(n_estimators=40, seed=1).fit(
            X, y, feature_names=["a", "b", "c"]
        )
        imp = model.feature_importance()
        self.assertEqual(set(imp), {"a", "b", "c"})
        self.assertGreater(sum(imp.values()), 0.0)

    def test_early_stopping_limits_rounds(self):
        X, y = _separable()
        model = GradientBoostingModel(
            n_estimators=200, patience=3, validation_fraction=0.2, seed=1
        ).fit(X, y)
        self.assertLessEqual(len(model._trees), 200)

    def test_subsample_is_deterministic(self):
        X, y = _separable()
        a = GradientBoostingModel(n_estimators=20, subsample=0.7, seed=8).fit(X, y)
        b = GradientBoostingModel(n_estimators=20, subsample=0.7, seed=8).fit(X, y)
        self.assertEqual(a.predict_proba(X), b.predict_proba(X))

    def test_nan_features_handled(self):
        X, y = _separable()
        Xn = [list(r) for r in X]
        for i in range(0, len(Xn), 5):
            Xn[i][0] = float("nan")
        model = GradientBoostingModel(n_estimators=30, seed=1).fit(Xn, y)
        probs = model.predict_proba(Xn)
        self.assertTrue(all(math.isfinite(p) for p in probs))

    def test_single_class_target(self):
        X, _ = _separable()
        zeros = GradientBoostingModel(n_estimators=10).fit(X, [0] * len(X))
        self.assertTrue(all(p == 0.0 for p in zeros.predict_proba(X)))
        ones = GradientBoostingModel(n_estimators=10).fit(X, [1] * len(X))
        self.assertTrue(all(p == 1.0 for p in ones.predict_proba(X)))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            GradientBoostingModel().predict_proba([[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
