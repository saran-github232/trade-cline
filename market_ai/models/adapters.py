"""Optional third-party model adapters and the ``build_model`` factory.

Third-party libraries (scikit-learn, XGBoost, LightGBM) are **optional**: they
are imported lazily inside the adapter methods and only when actually present.
``build_model`` always prefers a requested installed backend and otherwise
falls back to the pure-Python implementation, so importing this module never
requires any non-standard dependency.
"""

from __future__ import annotations

from importlib import util as _importlib_util
from typing import Any, Dict, List, Optional, Sequence

from ..config import DEFAULT_SEED
from .base import BaseModel
from .boosting import GradientBoostingModel
from .forest import RandomForestModel
from .linear import LogisticRegressionModel
from .trees import DecisionTreeModel

__all__ = [
    "available_backends",
    "SklearnAdapter",
    "XGBoostAdapter",
    "LightGBMAdapter",
    "build_model",
]


def _has_module(name: str) -> bool:
    """Return True when ``name`` is importable, without importing it."""
    try:
        return _importlib_util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - environment dependent
        return False


def available_backends() -> List[str]:
    """Return the optional third-party backends that are installed.

    The pure-Python implementations are always available and are therefore not
    listed here.
    """
    return [name for name in ("sklearn", "xgboost", "lightgbm") if _has_module(name)]


class _BackendAdapter(BaseModel):
    """Shared behaviour for third-party adapters: lazy import + clear errors.

    Subclasses set :attr:`module` (the import name) and :attr:`install_hint`
    (the pip package) and implement :meth:`fit` / :meth:`predict_proba`.
    """

    module: str = ""
    install_hint: str = ""
    _param_names = ("params", "seed")

    def __init__(self, params: Optional[Dict[str, Any]] = None, seed: int = DEFAULT_SEED, **kwargs: Any) -> None:
        super().__init__()
        self.seed = int(seed)
        merged: Dict[str, Any] = dict(params or {})
        merged.update(kwargs)
        self.params = merged
        self._estimator: Any = None

    @classmethod
    def is_available(cls) -> bool:
        """Return True when the backing library can be imported."""
        return _has_module(cls.module)

    def _require(self) -> Any:
        """Import and return the backing library, or raise ``RuntimeError``."""
        if not self.is_available():
            raise RuntimeError(
                f"{self.name} backend '{self.module}' is not installed; "
                f"install {self.install_hint} or use the pure-Python fallback"
            )
        return __import__(self.module)

    def _importance_from_estimator(self) -> Dict[str, float]:
        """Best-effort feature importance from a fitted sklearn-style model."""
        est = self._estimator
        values: List[float] = []
        if est is not None and hasattr(est, "feature_importances_"):
            values = [float(v) for v in est.feature_importances_]
        elif est is not None and hasattr(est, "coef_"):
            coef = est.coef_
            row = coef[0] if hasattr(coef, "__getitem__") and len(coef) else coef
            values = [abs(float(v)) for v in row]
        else:
            values = [0.0] * self._n_features
        names = self.feature_names or [f"f{i}" for i in range(len(values))]
        return {names[j]: values[j] for j in range(min(len(names), len(values)))}


class SklearnAdapter(_BackendAdapter):
    """Adapter around scikit-learn estimators (logistic / trees / ensembles).

    ``model`` selects the estimator family; extra ``params`` are forwarded to
    the estimator constructor.  Raises ``RuntimeError`` on use when
    scikit-learn is not installed.
    """

    name = "sklearn"
    module = "sklearn"
    install_hint = "'scikit-learn'"
    _param_names = ("model", "params", "seed")

    def __init__(
        self,
        model: str = "logistic",
        params: Optional[Dict[str, Any]] = None,
        seed: int = DEFAULT_SEED,
        **kwargs: Any,
    ) -> None:
        super().__init__(params=params, seed=seed, **kwargs)
        self.model = str(model)

    def _make_estimator(self) -> Any:
        """Construct the requested scikit-learn estimator (lazy import)."""
        self._require()
        extra = dict(self.params)
        if self.model in ("logistic", "logistic_regression"):
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000, random_state=self.seed, **extra)
        if self.model in ("random_forest", "forest"):
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(random_state=self.seed, **extra)
        if self.model in ("gradient_boosting", "boosting"):
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(random_state=self.seed, **extra)
        if self.model in ("decision_tree", "tree"):
            from sklearn.tree import DecisionTreeClassifier

            return DecisionTreeClassifier(random_state=self.seed, **extra)
        raise ValueError(f"unknown sklearn model {self.model!r}")

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "SklearnAdapter":
        """Fit the wrapped estimator; returns ``self``."""
        self._require()
        Xi, yb, w = self._fit_prepare(
            X,
            y,
            feature_names=feature_names,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        estimator = self._make_estimator()
        estimator.fit(Xi, yb, sample_weight=w)
        self._estimator = estimator
        self._fitted = True
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Return the wrapped estimator's class-1 probability."""
        Xp = self._prepare_predict(X)
        probs = self._estimator.predict_proba(Xp)
        return [float(row[1]) for row in probs]

    def feature_importance(self) -> Dict[str, float]:
        """Delegate to the estimator's coefficients or importances."""
        return self._importance_from_estimator()


class XGBoostAdapter(_BackendAdapter):
    """Adapter around ``xgboost.XGBClassifier``; lazy import, clear errors."""

    name = "xgboost"
    module = "xgboost"
    install_hint = "'xgboost'"

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "XGBoostAdapter":
        """Fit an ``XGBClassifier``; returns ``self``."""
        self._require()
        import xgboost as xgb

        Xi, yb, w = self._fit_prepare(
            X,
            y,
            feature_names=feature_names,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        estimator = xgb.XGBClassifier(random_state=self.seed, **self.params)
        estimator.fit(Xi, yb, sample_weight=w)
        self._estimator = estimator
        self._fitted = True
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Return the XGBoost class-1 probability."""
        Xp = self._prepare_predict(X)
        probs = self._estimator.predict_proba(Xp)
        return [float(row[1]) for row in probs]

    def feature_importance(self) -> Dict[str, float]:
        """Delegate to ``feature_importances_``."""
        return self._importance_from_estimator()


class LightGBMAdapter(_BackendAdapter):
    """Adapter around ``lightgbm.LGBMClassifier``; lazy import, clear errors."""

    name = "lightgbm"
    module = "lightgbm"
    install_hint = "'lightgbm'"

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        *,
        feature_names: Optional[Sequence[str]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "LightGBMAdapter":
        """Fit an ``LGBMClassifier``; returns ``self``."""
        self._require()
        import lightgbm as lgb

        Xi, yb, w = self._fit_prepare(
            X,
            y,
            feature_names=feature_names,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        estimator = lgb.LGBMClassifier(random_state=self.seed, **self.params)
        estimator.fit(Xi, yb, sample_weight=w)
        self._estimator = estimator
        self._fitted = True
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Return the LightGBM class-1 probability."""
        Xp = self._prepare_predict(X)
        probs = self._estimator.predict_proba(Xp)
        return [float(row[1]) for row in probs]

    def feature_importance(self) -> Dict[str, float]:
        """Delegate to ``feature_importances_``."""
        return self._importance_from_estimator()



#: Mapping of pure-Python model names (and friendly aliases) to their classes.
_PURE_MODELS = {
    "logistic": LogisticRegressionModel,
    "logistic_regression": LogisticRegressionModel,
    "decision_tree": DecisionTreeModel,
    "tree": DecisionTreeModel,
    "random_forest": RandomForestModel,
    "forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "boosting": GradientBoostingModel,
    "gbm": GradientBoostingModel,
}


def _filter_params(cls: type, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the params ``cls`` declares, so fallbacks never crash."""
    names = getattr(cls, "_param_names", None)
    if not names:
        return dict(params)
    return {key: value for key, value in params.items() if key in names}


def build_model(name: str, **params: Any) -> BaseModel:
    """Construct the best available model for ``name``.

    Supported names: ``"logistic"``, ``"decision_tree"``, ``"random_forest"``,
    ``"gradient_boosting"``, ``"xgboost"`` and ``"lightgbm"`` (plus a few
    friendly aliases).  ``xgboost``/``lightgbm`` use the installed third-party
    backend when present and otherwise fall back to the pure-Python
    :class:`GradientBoostingModel`, filtering ``params`` to those it accepts so
    the fallback never fails on backend-specific arguments.
    """
    key = str(name).strip().lower()
    if key in ("xgboost", "xgb"):
        if XGBoostAdapter.is_available():
            return XGBoostAdapter(**params)
        return GradientBoostingModel(**_filter_params(GradientBoostingModel, params))
    if key in ("lightgbm", "lgbm", "lgb"):
        if LightGBMAdapter.is_available():
            return LightGBMAdapter(**params)
        return GradientBoostingModel(**_filter_params(GradientBoostingModel, params))
    cls = _PURE_MODELS.get(key)
    if cls is None:
        raise ValueError(
            f"unknown model name {name!r}; expected one of "
            "logistic, decision_tree, random_forest, gradient_boosting, "
            "xgboost, lightgbm"
        )
    return cls(**_filter_params(cls, params))

