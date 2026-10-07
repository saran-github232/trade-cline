# INTERFACES.md - Frozen internal contracts

This file is the coordination contract between subsystems. **Do not change a
signature listed here without updating this file**, because other modules are
written against it in parallel.

All code is **Python 3.11 standard library only**. No numpy, pandas, sklearn,
requests, matplotlib. If an optional third-party library is genuinely useful,
it must be imported lazily inside a `try/except ImportError` and the pure
Python path must remain the default.

Everything imports its core types from `market_ai.types` and configuration
from `market_ai.config`.

---

## 0. Already implemented (do not rewrite)

* `market_ai/types.py` - `Candle`, `MarketSeries`, `FeatureVector`, `Signal`,
  `ProbabilitySet`, `Prediction`, `Outcome`, `Experience`, enums
  (`Timeframe`, `Direction`, `Decision`, `Regime`, `ErrorCategory`,
  `DataQuality`), and time helpers (`to_utc_epoch`, `iso_to_epoch`,
  `epoch_to_iso`).
* `market_ai/config.py` - `Config` plus sub-configs (`DataConfig`,
  `RegimeConfig`, `EnsembleConfig`, `DecisionConfig`, `RiskConfig`,
  `BacktestConfig`, `PromotionConfig`, `VisionConfig`, `ReasoningConfig`,
  `StorageConfig`, `APIConfig`), `load_config()`.
* `market_ai/utils/stats.py` - `mean`, `median`, `stdev`, `percentile`,
  `quantile`, `zscore`, `safe_div`, `clamp`, `correlation`,
  `wilson_interval`, `normal_cdf`, `binomial_p_value`,
  `expected_calibration_error`, `brier_score`, `log_loss`, `profit_factor`,
  `max_drawdown`, `longest_losing_streak`, `expectancy`, `sharpe_like`,
  `accuracy`, `precision`, `recall`, `f1_score`, `contingency`.
* `market_ai/utils/logging.py` - `get_logger(name)`, `log_event(logger,
  level, message, **ctx)`, `timed(logger, label, **ctx)`, `scrub(payload)`.
* `market_ai/utils/jsonio.py` - `dump_json`, `load_json`, `dumps`, `loads`,
  `atomic_write_text`, `atomic_write_json` (NaN/Infinity-safe).
* `market_ai/data/*` - `DataProvider` protocol, `HistoricalDataProvider`,
  `ReplayDataProvider`, `SyntheticProvider`, `CsvProvider`, and
  `validate_series()` returning a `DataQualityReport`.
* `market_ai/features/indicators.py` - all indicator functions.
* `market_ai/storage/db.py` - SQLite `Database` with repositories.

---

## 1. Indicators (`market_ai.features.indicators`)

All functions take a `Sequence[float]` (or OHLC sequences) and return
`List[float]` of the **same length** as the input, with `float('nan')`
padding at the front where the indicator is not yet defined. This makes
index alignment with candles trivial and leakage-testable.

```python
sma(values, period) -> List[float]
ema(values, period) -> List[float]
wilder_smooth(values, period) -> List[float]
rsi(closes, period=14) -> List[float]
macd(closes, fast=12, slow=26, signal=9) -> Dict[str, List[float]]  # macd, signal, hist
stochastic(highs, lows, closes, k=14, d=3) -> Dict[str, List[float]]  # k, d
roc(closes, period=10) -> List[float]
atr(highs, lows, closes, period=14) -> List[float]
bollinger(closes, period=20, mult=2.0) -> Dict[str, List[float]]  # mid, upper, lower, width, pct_b
rolling_volatility(closes, period=20, annualize=False) -> List[float]
rolling_max(values, period) -> List[float]
rolling_min(values, period) -> List[float]
vwap(highs, lows, closes, volumes, period=20) -> List[float]
adx(highs, lows, closes, period=14) -> Dict[str, List[float]]  # adx, plus_di, minus_di
slope(values, period=10) -> List[float]   # normalised least-squares slope
```

---

## 2. Market data (`market_ai.data`)

```python
class DataProvider(Protocol):
    def get_series(self, asset: str, timeframe: Timeframe,
                   start: int | None = None,
                   end: int | None = None) -> MarketSeries: ...

class HistoricalDataProvider:      # wraps a static store of MarketSeries
    def __init__(self, series: Iterable[MarketSeries]): ...
    def get_series(self, asset, timeframe, start=None, end=None) -> MarketSeries

class ReplayDataProvider:          # deterministic, leakage-free cursor
    def __init__(self, provider: DataProvider, *,
                 horizon_bars: int = 3, warmup_bars: int = 250): ...
    def __iter__(self) -> Iterator[ReplayStep]   # ascending time
    def steps(self) -> Iterator[ReplayStep]

@dataclass(frozen=True)
class ReplayStep:
    asset: str
    timeframe: Timeframe
    index: int                  # index of the decision bar in the full series
    as_of: int                  # close time of the decision bar
    history: MarketSeries       # ONLY candles with close_time <= as_of
    entry_price: float          # close of the decision bar
    future: MarketSeries        # ONLY used by the outcome resolver
    horizon_bars: int

@dataclass
class DataQualityReport:
    asset: str
    timeframe: Timeframe
    quality: DataQuality
    issues: List[str]
    missing_bars: int
    duplicate_timestamps: int
    invalid_ohlc: int
    out_of_order: int
    outliers: int
    coverage: float
    def to_dict(self) -> Dict[str, Any]
```

`ReplayDataProvider` **must never** place a candle whose `close_time > as_of`
inside `ReplayStep.history`. This is the system's leakage firewall.

---

## 3. Features (`market_ai.features`)

```python
# price_action.py
def analyse_price_action(series: MarketSeries) -> Dict[str, float]   # feature dict
def detect_patterns(series: MarketSeries) -> Dict[str, Any]          # named booleans

# structure.py
def analyse_structure(series: MarketSeries, *, lookback: int = 60) -> Dict[str, float]
def find_swings(highs, lows, lookback=3) -> Tuple[List[int], List[int]]
def support_resistance_zones(series, lookback=120, tolerance=0.002) -> Dict[str, Any]

# feature_engine.py
FEATURE_VERSION: str
class FeatureEngine:
    def __init__(self, config: Config | None = None): ...
    def compute(self, series: MarketSeries) -> FeatureVector   # series must already be truncated
    def feature_names(self) -> List[str]
    def version(self) -> str
```

Feature keys use dotted namespaces: `trend.ema_9`, `momentum.rsi_14`,
`volatility.atr_14`, `pa.body_ratio`, `structure.hh`, `regime.*`.

---

## 4. Regime (`market_ai.regime`)

```python
@dataclass(frozen=True)
class RegimeResult:
    regime: Regime
    confidence: float
    supporting_features: Mapping[str, float]
    timestamp: int
    def to_dict(self) -> Dict[str, Any]

class RegimeDetector:
    def __init__(self, config: Config | None = None): ...
    def detect(self, series: MarketSeries, features: FeatureVector | None = None) -> RegimeResult
```

---

## 5. Multi-timeframe (`market_ai.multi_timeframe`)

```python
@dataclass(frozen=True)
class TimeframeView:
    timeframe: Timeframe
    direction: Direction
    strength: float
    detail: Mapping[str, Any]

@dataclass(frozen=True)
class MultiTimeframeResult:
    higher: TimeframeView | None
    middle: TimeframeView | None
    lower: TimeframeView | None
    agreement: float          # 0..1
    conflict: float           # 0..1
    aligned_direction: Direction
    timestamp: int
    def to_dict(self) -> Dict[str, Any]

class MultiTimeframeAnalyzer:
    def __init__(self, config: Config | None = None): ...
    def analyse(self, views: Mapping[Timeframe, MarketSeries],
                as_of: int, decision_timeframe: Timeframe) -> MultiTimeframeResult
```

The lower timeframe may never override the higher-timeframe regime; conflict
increases NO-TRADE probability instead.

---

## 6. Models (`market_ai.models`)

```python
# base.py
class Model(Protocol):
    name: str
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int],
            *, feature_names: Sequence[str] | None = None,
            sample_weight: Sequence[float] | None = None) -> "Model"
    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]  # P(class=1)
    def feature_importance(self) -> Dict[str, float]
    def get_params(self) -> Dict[str, Any]
    def set_params(self, **params) -> "Model"

def standardize_fit(X) -> Tuple[List[float], List[float]]      # means, stds
def standardize_apply(X, means, stds) -> List[List[float]]
def impute(X, strategy="median") -> List[List[float]]

# linear.py      -> LogisticRegressionModel
# trees.py       -> DecisionTreeModel
# forest.py      -> RandomForestModel
# boosting.py    -> GradientBoostingModel
# adapters.py    -> SklearnAdapter, XGBoostAdapter, LightGBMAdapter,
#                   available_backends() -> List[str], build_model(name, **params)
# calibration.py -> Calibrator (Platt scaling + isotonic), calibration_report()
# registry.py    -> see section 10
```

`build_model(name, **params)` accepts: `"logistic"`, `"random_forest"`,
`"gradient_boosting"`, `"decision_tree"`, `"xgboost"`, `"lightgbm"` and
returns the best available implementation, preferring an installed
third-party backend when explicitly requested by name.

All models are pure-Python and deterministic given a seed.

---

## 7. Vision (`market_ai.vision`)

```python
# schema.py
VISION_RESPONSE_SCHEMA: Dict[str, Any]
class VisionResult:
    direction_bias: Direction          # UP | DOWN | NEUTRAL
    trend: str
    structure: str
    support: List[float]
    resistance: List[float]
    breakout_probability: float
    reversal_probability: float
    momentum: str
    volatility: str
    confidence: float
    invalidating_conditions: List[str]
    reasoning_summary: str
    provider: str
    degraded: bool                     # True when a fallback was used
    raw: Mapping[str, Any]
    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, payload) -> "VisionResult"

def validate_vision_payload(payload: Mapping[str, Any]) -> Tuple[bool, List[str]]

# analyser.py
class ChartVisionAnalyser:
    def __init__(self, config: Config | None = None): ...
    def analyse(self, *, series: MarketSeries, summary: Mapping[str, Any],
                image_path: str | None = None,
                timeframe: Timeframe | None = None,
                asset: str | None = None) -> VisionResult
```

`provider="offline"` is the default and MUST work with no network and no API
key. External providers are optional, lazily imported, read credentials from
environment variables, and fall back to `offline` on any failure while
setting `degraded=True`.

---

## 8. Reasoning supervisor (`market_ai.reasoning`)

```python
@dataclass(frozen=True)
class ReasoningResult:
    recommendation: Decision
    contradictions: List[str]
    weak_evidence: List[str]
    invalidating_conditions: List[str]
    abnormal_conditions: List[str]
    explanation: str
    confidence_adjustment: float      # in [-1, 0]; may only reduce confidence
    provider: str
    degraded: bool
    raw: Mapping[str, Any]
    def to_dict(self) -> Dict[str, Any]

class ReasoningSupervisor:
    def __init__(self, config: Config | None = None): ...
    def review(self, *, context: Mapping[str, Any]) -> ReasoningResult
```

The supervisor may **only** push towards NO-TRADE or lower confidence. It can
never invent market data, change weights, change risk limits, or create a
directional call.

---

## 9. Storage (`market_ai.storage.db`)

```python
class Database:
    def __init__(self, path: str | Path): ...
    def initialize(self) -> None
    def close(self) -> None
    # repositories are plain attributes:
    #   db.predictions, db.outcomes, db.experiences, db.models,
    #   db.backtests, db.datasets, db.events, db.paper_trades
```

Every repository exposes `add(obj)`, `get(id)`, `list(**filters)`,
`count(**filters)`, `delete(id)`. Rows store JSON for nested payloads.

---

## 10. Model registry (`market_ai.models.registry`)

```python
@dataclass
class ModelRecord:
    model_version: str
    role: str                  # "champion" | "challenger" | "archived"
    model_name: str
    params: Dict[str, Any]
    feature_version: str
    dataset_version: str
    training_period: Tuple[int, int]
    validation_period: Tuple[int, int]
    test_period: Tuple[int, int]
    metrics: Dict[str, Any]
    calibration: Dict[str, Any]
    created_at: int
    artifact_path: str
    notes: str
    def to_dict(self) -> Dict[str, Any]

class ModelRegistry:
    def __init__(self, root: str | Path): ...
    def save(self, record: ModelRecord, model: Any) -> str
    def load(self, model_version: str) -> Any
    def get(self, model_version: str) -> ModelRecord
    def list(self) -> List[ModelRecord]
    def champion(self) -> ModelRecord | None
    def set_champion(self, model_version: str) -> None
    def rollback(self, to_version: str | None = None) -> ModelRecord | None
    def history(self) -> List[Dict[str, Any]]
```

Artifacts are JSON + a `.json` model blob so no pickle is required and no
arbitrary code executes on load.

---

## 11. Backtest (`market_ai.backtest`)

```python
# metrics.py
@dataclass
class BacktestMetrics: ...   # see below

def compute_metrics(records: Sequence[Mapping[str, Any]], *, payout: float) -> BacktestMetrics
def group_metrics(records, key_fn, *, payout: float) -> Dict[str, BacktestMetrics]

# engine.py
class BacktestEngine:
    def __init__(self, config: Config | None = None, *, payout: float | None = None): ...
    def run(self, series: MarketSeries, *, strategy_factory=None,
            horizons: Sequence[int] = ()) -> BacktestResult

@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    records: List[Dict[str, Any]]          # one per decision bar
    by_regime: Dict[str, BacktestMetrics]
    by_confidence: Dict[str, BacktestMetrics]
    by_timeframe: Dict[str, BacktestMetrics]
    by_asset: Dict[str, BacktestMetrics]
    by_variant: Dict[str, BacktestMetrics] # baseline/technical/ml/vision/ensemble
    equity_curve: List[float]
    drawdown_curve: List[float]
    leakage_report: Dict[str, Any]
    overfitting_report: Dict[str, Any]
    folds: List[Dict[str, Any]]
    def to_dict(self) -> Dict[str, Any]

# walkforward.py
class WalkForwardSplitter:
    def __init__(self, train, validation, test, embargo, step=None): ...
    def split(self, n: int) -> Iterator[Tuple[slice, slice, slice]]

# report.py
def write_reports(result: BacktestResult, out_dir: str | Path) -> Dict[str, str]
```

`BacktestMetrics` fields: `total_predictions`, `up_predictions`,
`down_predictions`, `no_trade_predictions`, `wins`, `losses`, `win_rate`,
`win_rate_ci`, `up_precision`, `down_precision`, `recall`, `expectancy`,
`profit_factor`, `max_drawdown`, `max_drawdown_pct`, `longest_losing_streak`,
`average_win`, `average_loss`, `calibration_error`, `brier`, `log_loss`,
`total_pnl`, `final_equity`, `sample_size_ok`, `sharpe_like`, plus
`to_dict()`.

---

## 12. Outcomes / Experience / Training / Retraining / Risk / Paper trading

```python
# outcomes/resolver.py
class OutcomeResolver:
    def __init__(self, config=None): ...
    def resolve(self, prediction: Prediction, series: MarketSeries, *,
                payout: float | None = None) -> Outcome | None
    def resolve_pending(self, predictions: Sequence[Prediction],
                        series_by_key: Mapping[Tuple[str, Timeframe], MarketSeries]) -> List[Outcome]

def classify_outcome(decision: Decision, actual: Direction) -> ErrorCategory

# experience/store.py
class ExperienceStore:
    def __init__(self, db=None, config=None): ...
    def add(self, experience: Experience) -> None
    def add_many(self, items) -> int
    def all(self) -> List[Experience]
    def count(self) -> int
    def since(self, ts: int) -> List[Experience]

# experience/analytics.py
def error_breakdown(experiences) -> Dict[str, Any]
def regime_failures(experiences) -> Dict[str, Any]
def confidence_reliability(experiences) -> Dict[str, Any]
def recurring_patterns(experiences) -> List[Dict[str, Any]]
def tag_experience(experience, context) -> Tuple[str, ...]
```

---

## 13. Conventions

* Public functions have docstrings; non-obvious logic has inline comments.
* Never raise on bad *data* - return a degraded/NO-TRADE result instead.
  Do raise `ValueError` on programmer error (bad types, bad shapes).
* Every module gets a `tests/test_<module>.py` using `unittest`.
* Run `python3 -m unittest discover -s tests -v` before declaring done.
* No network access in tests. No hard-coded credentials. No `pickle`.
* Do not edit files outside your assigned ownership list.
