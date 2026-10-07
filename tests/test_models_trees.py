"""Tests for the pure-Python CART decision tree."""

import math
import random
import unittest

from market_ai.models.trees import DecisionTreeModel


def _separable(n=300, d=4, seed=0):
    """Synthetic data with an axis-aligned-ish separable structure."""
    rng = random.Random(seed)
    X = []
    y = []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.0) for _ in range(d)]
        score = row[0] + 0.5 * row[1]
        X.append(row)
        y.append(1 if score > 0.0 else 0)
    return X, y


class DecisionTreeTests(unittest.TestCase):
    def test_learns_separable_data(self):
        X, y = _separable()
        model = DecisionTreeModel(max_depth=4).fit(X, y)
        probs = model.predict_proba(X)
        acc = sum(1 for p, t in zip(probs, y) if (p >= 0.5) == (t == 1)) / len(y)
        self.assertGreater(acc, 0.9)

    def test_probabilities_in_unit_interval(self):
        X, y = _separable()
        probs = DecisionTreeModel(max_depth=3).fit(X, y).predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_determinism_same_seed(self):
        X, y = _separable()
        a = DecisionTreeModel(seed=11).fit(X, y).predict_proba(X)
        b = DecisionTreeModel(seed=11).fit(X, y).predict_proba(X)
        self.assertEqual(a, b)

    def test_feature_importance_sums_positive(self):
        X, y = _separable()
        model = DecisionTreeModel(max_depth=4).fit(X, y, feature_names=["a", "b", "c", "d"])
        imp = model.feature_importance()
        self.assertEqual(set(imp), {"a", "b", "c", "d"})
        self.assertGreater(sum(imp.values()), 0.0)
        self.assertGreater(imp["a"], 0.0)

    def test_entropy_criterion(self):
        X, y = _separable()
        model = DecisionTreeModel(criterion="entropy", max_depth=4).fit(X, y)
        probs = model.predict_proba(X)
        acc = sum(1 for p, t in zip(probs, y) if (p >= 0.5) == (t == 1)) / len(y)
        self.assertGreater(acc, 0.85)

    def test_max_features_variants(self):
        X, y = _separable()
        for mf in ("sqrt", "log2", 0.5, 2, None):
            model = DecisionTreeModel(max_features=mf, seed=2).fit(X, y)
            probs = model.predict_proba(X)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_depth_limits_tree(self):
        X, y = _separable()
        shallow = DecisionTreeModel(max_depth=1).fit(X, y)
        deep = DecisionTreeModel(max_depth=6).fit(X, y)
        # A depth-1 stump cannot overfit as well as a deep tree on train data.
        shallow_acc = sum(
            1 for p, t in zip(shallow.predict_proba(X), y) if (p >= 0.5) == (t == 1)
        ) / len(y)
        deep_acc = sum(
            1 for p, t in zip(deep.predict_proba(X), y) if (p >= 0.5) == (t == 1)
        ) / len(y)
        self.assertGreaterEqual(deep_acc, shallow_acc)

    def test_nan_features_handled(self):
        X, y = _separable()
        Xn = [list(r) for r in X]
        for i in range(0, len(Xn), 4):
            Xn[i][2] = float("nan")
        model = DecisionTreeModel(max_depth=4).fit(Xn, y)
        probs = model.predict_proba(Xn)
        self.assertTrue(all(math.isfinite(p) for p in probs))

    def test_single_class_target(self):
        X, _ = _separable()
        zeros = DecisionTreeModel().fit(X, [0] * len(X))
        self.assertTrue(all(p == 0.0 for p in zeros.predict_proba(X)))
        ones = DecisionTreeModel().fit(X, [1] * len(X))
        self.assertTrue(all(p == 1.0 for p in ones.predict_proba(X)))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            DecisionTreeModel().predict_proba([[1.0, 2.0]])

    def test_get_set_params(self):
        model = DecisionTreeModel(max_depth=3)
        self.assertEqual(model.get_params()["max_depth"], 3)
        model.set_params(max_depth=5)
        self.assertEqual(model.max_depth, 5)


if __name__ == "__main__":
    unittest.main()
