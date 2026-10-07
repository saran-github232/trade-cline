"""Tests for the third-party adapters and the ``build_model`` factory."""

import unittest

from market_ai.models.adapters import (
    LightGBMAdapter,
    SklearnAdapter,
    XGBoostAdapter,
    available_backends,
    build_model,
)
from market_ai.models.boosting import GradientBoostingModel
from market_ai.models.forest import RandomForestModel
from market_ai.models.linear import LogisticRegressionModel
from market_ai.models.trees import DecisionTreeModel


def _data(n=120, seed=0):
    import random

    rng = random.Random(seed)
    X = []
    y = []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.0) for _ in range(3)]
        X.append(row)
        y.append(1 if row[0] + row[1] > 0.0 else 0)
    return X, y


class AvailableBackendsTests(unittest.TestCase):
    def test_returns_list_subset(self):
        backends = available_backends()
        self.assertIsInstance(backends, list)
        for name in backends:
            self.assertIn(name, ("sklearn", "xgboost", "lightgbm"))

    def test_is_available_consistent(self):
        backends = set(available_backends())
        self.assertEqual(SklearnAdapter.is_available(), "sklearn" in backends)
        self.assertEqual(XGBoostAdapter.is_available(), "xgboost" in backends)
        self.assertEqual(LightGBMAdapter.is_available(), "lightgbm" in backends)


class BuildModelTests(unittest.TestCase):
    def test_pure_names(self):
        self.assertIsInstance(build_model("logistic"), LogisticRegressionModel)
        self.assertIsInstance(build_model("decision_tree"), DecisionTreeModel)
        self.assertIsInstance(build_model("random_forest"), RandomForestModel)
        self.assertIsInstance(build_model("gradient_boosting"), GradientBoostingModel)

    def test_params_forwarded(self):
        model = build_model("random_forest", n_estimators=7, seed=2)
        self.assertEqual(model.get_params()["n_estimators"], 7)

    def test_xgboost_fallback_when_missing(self):
        model = build_model("xgboost", n_estimators=12, seed=2)
        if XGBoostAdapter.is_available():
            self.assertIsInstance(model, XGBoostAdapter)
        else:
            self.assertIsInstance(model, GradientBoostingModel)
            self.assertEqual(model.get_params()["n_estimators"], 12)

    def test_lightgbm_fallback_when_missing(self):
        model = build_model("lightgbm", n_estimators=12, seed=2)
        if LightGBMAdapter.is_available():
            self.assertIsInstance(model, LightGBMAdapter)
        else:
            self.assertIsInstance(model, GradientBoostingModel)

    def test_fallback_filters_unknown_params(self):
        # ``num_leaves`` is LightGBM-specific; the pure-Python fallback must
        # not choke on it.
        model = build_model("lightgbm", num_leaves=31, n_estimators=10)
        self.assertIsInstance(model, GradientBoostingModel)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            build_model("does_not_exist")

    def test_built_model_fits_and_predicts(self):
        X, y = _data()
        model = build_model("gradient_boosting", n_estimators=10, seed=1)
        model.fit(X, y)
        probs = model.predict_proba(X)
        self.assertEqual(len(probs), len(X))


class AdapterErrorTests(unittest.TestCase):
    def test_unavailable_adapters_raise_runtime_error(self):
        X, y = _data()
        for adapter in (SklearnAdapter(), XGBoostAdapter(), LightGBMAdapter()):
            if adapter.is_available():
                continue  # installed backend: nothing to assert here
            with self.assertRaises(RuntimeError):
                adapter.fit(X, y)

    def test_runtime_error_message_is_clear(self):
        if XGBoostAdapter.is_available():
            self.skipTest("xgboost installed")
        with self.assertRaises(RuntimeError) as ctx:
            XGBoostAdapter().fit(*_data())
        self.assertIn("xgboost", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
