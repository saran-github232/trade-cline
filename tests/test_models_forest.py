"""Tests for the pure-Python random forest."""

import math
import random
import unittest

from market_ai.models.forest import RandomForestModel


def _separable(n=400, d=4, seed=0):
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


class RandomForestTests(unittest.TestCase):
    def test_learns_separable_data(self):
        X, y = _separable()
        model = RandomForestModel(n_estimators=40, seed=1).fit(X, y)
        probs = model.predict_proba(X)
        acc = sum(1 for p, t in zip(probs, y) if (p >= 0.5) == (t == 1)) / len(y)
        self.assertGreater(acc, 0.9)

    def test_probabilities_in_unit_interval(self):
        X, y = _separable()
        probs = RandomForestModel(n_estimators=20, seed=1).fit(X, y).predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_determinism_same_seed(self):
        X, y = _separable()
        a = RandomForestModel(n_estimators=15, seed=9).fit(X, y).predict_proba(X)
        b = RandomForestModel(n_estimators=15, seed=9).fit(X, y).predict_proba(X)
        self.assertEqual(a, b)

    def test_oob_probabilities_and_score(self):
        X, y = _separable()
        model = RandomForestModel(n_estimators=40, seed=3).fit(X, y)
        oob = model.oob_probabilities()
        self.assertEqual(len(oob), len(X))
        score = model.oob_score()
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertGreater(score, 0.7)

    def test_feature_importance_sums_positive(self):
        X, y = _separable()
        model = RandomForestModel(n_estimators=20, seed=1).fit(
            X, y, feature_names=["a", "b", "c", "d"]
        )
        imp = model.feature_importance()
        self.assertEqual(set(imp), {"a", "b", "c", "d"})
        self.assertGreater(sum(imp.values()), 0.0)

    def test_n_jobs_param_accepted(self):
        X, y = _separable()
        model = RandomForestModel(n_estimators=10, n_jobs=4, seed=1).fit(X, y)
        self.assertEqual(model.get_params()["n_jobs"], 4)
        self.assertEqual(len(model.predict_proba(X)), len(X))

    def test_nan_features_handled(self):
        X, y = _separable()
        Xn = [list(r) for r in X]
        for i in range(0, len(Xn), 6):
            Xn[i][1] = float("nan")
        model = RandomForestModel(n_estimators=15, seed=1).fit(Xn, y)
        probs = model.predict_proba(Xn)
        self.assertTrue(all(math.isfinite(p) for p in probs))

    def test_single_class_target(self):
        X, _ = _separable()
        zeros = RandomForestModel(n_estimators=5).fit(X, [0] * len(X))
        self.assertTrue(all(p == 0.0 for p in zeros.predict_proba(X)))
        ones = RandomForestModel(n_estimators=5).fit(X, [1] * len(X))
        self.assertTrue(all(p == 1.0 for p in ones.predict_proba(X)))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            RandomForestModel().predict_proba([[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
