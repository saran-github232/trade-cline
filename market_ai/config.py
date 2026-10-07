"""Central configuration for the market-analysis system.

Everything tunable lives here so that backtests, paper trading and the live
prediction path share exactly one source of truth.  Configuration is a plain
frozen dataclass built from (in order of precedence):

    explicit kwargs  >  environment variables  >  defaults

No secrets are ever hard-coded.  Optional third-party integrations read their
credentials from the environment; when absent the system falls back to the
deterministic offline components rather than failing.

Version strings are *behavioural* contracts.  If you change a feature
definition, bump ``FEATURE_VERSION``; if you change ensemble/gate logic, bump
``STRATEGY_VERSION``.  Backtest reports and model registry entries record
these so that results can never be silently compared across incompatible
code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .types import Timeframe

# --------------------------------------------------------------------------
# Behavioural version contracts
# --------------------------------------------------------------------------
FEATURE_VERSION = "feat-1.0.0"
STRATEGY_VERSION = "strat-1.0.0"
ENSEMBLE_VERSION = "ens-1.0.0"
GATE_VERSION = "gate-1.0.0"
REGIME_VERSION = "regime-1.0.0"
DATASET_VERSION = "ds-1.0.0"
DEFAULT_SEED = 1337


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on", "y")


def _env_list(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = _env_str(name)
    if raw is None:
        return default
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    return parts or default


# --------------------------------------------------------------------------
# Configuration objects
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Market-data acquisition, validation and replay settings."""

    assets: Tuple[str, ...] = ("EURUSD", "GBPUSD", "BTCUSD")
    timeframes: Tuple[Timeframe, ...] = (
        Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1,
    )
    min_history_bars: int = 250
    max_gap_bars: int = 1
    outlier_z_threshold: float = 8.0
    max_missing_fraction: float = 0.02
    duplicate_tolerance_seconds: int = 0
    synthetic_bars: int = 6000
    synthetic_seed: int = DEFAULT_SEED
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @classmethod
    def from_env(cls) -> "DataConfig":
        return cls(
            assets=_env_list("MARKET_AI_ASSETS", ("EURUSD", "GBPUSD", "BTCUSD")),
            timeframes=tuple(
                Timeframe.parse(t)
                for t in _env_list("MARKET_AI_TIMEFRAMES", ("1m", "5m", "15m", "1h"))
            ),
            synthetic_bars=_env_int("MARKET_AI_SYNTHETIC_BARS", 6000),
            synthetic_seed=_env_int("MARKET_AI_SEED", DEFAULT_SEED),
            data_dir=Path(_env_str("MARKET_AI_DATA_DIR", "data") or "data"),
        )


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds used by the regime detector."""

    adx_trend_threshold: float = 22.0
    adx_strong_threshold: float = 32.0
    atr_low_percentile: float = 0.25
    atr_high_percentile: float = 0.80
    range_compression_ratio: float = 0.55
    breakout_lookback: int = 20
    slope_lookback: int = 10
    min_confidence: float = 0.35


@dataclass(frozen=True)
class EnsembleConfig:
    """Signal weights and disagreement handling.

    Weights are *initialisation values only*.  ``ValidationWeightOptimizer``
    measures whether different weights generalise better on unseen data, and
    the promoted weight vector is stored in the model registry.
    """

    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "technical": 0.20,
            "price_action": 0.20,
            "structure": 0.15,
            "regime": 0.15,
            "ml": 0.20,
            "vision": 0.10,
        }
    )
    mtf_modifier_strength: float = 0.25
    reasoning_modifier_strength: float = 0.15
    min_agreement: float = 0.45
    max_disagreement: float = 0.55

    @classmethod
    def from_env(cls) -> "EnsembleConfig":
        return cls()


@dataclass(frozen=True)
class DecisionConfig:
    """Thresholds for the final decision gate."""

    min_confidence: float = 0.55
    min_probability_margin: float = 0.08
    min_agreement: float = 0.45
    min_feature_coverage: float = 0.70
    min_historical_support: int = 30
    volatility_percentile_floor: float = 0.02
    volatility_percentile_ceiling: float = 0.985
    min_directional_precision: float = 0.52
    trusted_confidence_floor: float = 0.55
    cold_start_samples: int = 50

    @classmethod
    def from_env(cls) -> "DecisionConfig":
        return cls(
            min_confidence=_env_float("MARKET_AI_MIN_CONFIDENCE", 0.55),
            min_agreement=_env_float("MARKET_AI_MIN_AGREEMENT", 0.45),
        )


@dataclass(frozen=True)
class RiskConfig:
    """Paper-trading risk limits.  Martingale is explicitly unsupported."""

    starting_balance: float = 1000.0
    #: Fixed fraction of balance risked per trade.  Never increased after a
    #: loss - loss-chasing (Martingale) is not implemented anywhere.
    risk_per_trade: float = 0.01
    max_stake_fraction: float = 0.05
    daily_loss_limit_fraction: float = 0.05
    max_consecutive_losses: int = 6
    max_daily_trades: int = 40
    #: Payout assumption used for expectancy.  Deliberately conservative.
    payout: float = 0.85
    pessimistic_payout: float = 0.70

    @property
    def breakeven_win_rate(self) -> float:
        """Win rate required to break even at this payout."""
        return 1.0 / (1.0 + self.payout)

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            starting_balance=_env_float("MARKET_AI_STARTING_BALANCE", 1000.0),
            payout=_env_float("MARKET_AI_PAYOUT", 0.85),
            risk_per_trade=_env_float("MARKET_AI_RISK_PER_TRADE", 0.01),
        )


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward backtesting configuration."""

    train_window: int = 1200
    validation_window: int = 300
    test_window: int = 300
    embargo_bars: int = 30
    horizon_bars: int = 3
    min_samples_for_significance: int = 200
    permutation_trials: int = 5
    cooldown_bars: int = 1
    seed: int = DEFAULT_SEED

    @classmethod
    def from_env(cls) -> "BacktestConfig":
        return cls(
            train_window=_env_int("MARKET_AI_TRAIN_WINDOW", 1200),
            validation_window=_env_int("MARKET_AI_VALIDATION_WINDOW", 300),
            test_window=_env_int("MARKET_AI_TEST_WINDOW", 300),
            horizon_bars=_env_int("MARKET_AI_HORIZON_BARS", 3),
        )


@dataclass(frozen=True)
class PromotionConfig:
    """Champion/challenger promotion criteria (Phase 10).

    A challenger replaces the champion only if *every* gate passes.  Training
    accuracy is not among the gates, by design.
    """

    min_new_samples: int = 300
    min_out_of_sample_samples: int = 200
    min_expectancy_improvement: float = 0.0
    min_precision_improvement: float = 0.005
    min_oos_precision: float = 0.52
    min_oos_expectancy: float = 0.0
    max_calibration_error: float = 0.10
    min_regime_precision: float = 0.45
    max_fold_degradation: float = 0.15
    max_drawdown: float = 0.25
    max_p_value: float = 0.10
    min_oos_span_bars: int = 200
    retrain_every_n_samples: int = 1000

    @classmethod
    def from_env(cls) -> "PromotionConfig":
        return cls()


@dataclass(frozen=True)
class VisionConfig:
    """Chart-vision settings.

    ``provider`` selects the implementation:

    ``"offline"``
        Deterministic, dependency-free analyser that consumes the structured
        numerical summary.  Always available; this is the default so the
        system works with no network and no API key.
    ``"openai"`` / ``"anthropic"``
        Optional multimodal providers.  They require credentials from the
        environment and a caller-supplied image.  When they fail, the system
        falls back to ``offline`` and records that it did so.
    """

    provider: str = "offline"
    fallback_provider: str = "offline"
    timeout_seconds: float = 20.0
    max_retries: int = 2
    #: Vision is evidence, never authority: cap how much it can move the gate.
    max_influence: float = 0.10
    max_self_reported_confidence: float = 0.95
    image_dir: Path = field(default_factory=lambda: Path("artifacts/charts"))

    @classmethod
    def from_env(cls) -> "VisionConfig":
        return cls(
            provider=_env_str("MARKET_AI_VISION_PROVIDER", "offline") or "offline",
            fallback_provider=_env_str("MARKET_AI_VISION_FALLBACK", "offline") or "offline",
            timeout_seconds=_env_float("MARKET_AI_VISION_TIMEOUT", 20.0),
        )


@dataclass(frozen=True)
class ReasoningConfig:
    """AI reasoning supervisor settings.

    The supervisor explains and flags contradictions.  It can *only* push the
    decision towards NO-TRADE; it can never create a directional call that the
    quantitative evidence did not already support.
    """

    provider: str = "offline"
    timeout_seconds: float = 25.0
    max_retries: int = 2
    max_confidence_multiplier: float = 1.05
    hard_contradiction_forces_no_trade: bool = True

    @classmethod
    def from_env(cls) -> "ReasoningConfig":
        return cls(provider=_env_str("MARKET_AI_REASONING_PROVIDER", "offline") or "offline")


@dataclass(frozen=True)
class StorageConfig:
    """SQLite-backed storage settings (stdlib only, no server required)."""

    db_path: Path = field(default_factory=lambda: Path("artifacts/market_ai.db"))
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    journal_mode: str = "WAL"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        return cls(
            db_path=Path(_env_str("MARKET_AI_DB", "artifacts/market_ai.db") or "artifacts/market_ai.db"),
            artifacts_dir=Path(_env_str("MARKET_AI_ARTIFACTS", "artifacts") or "artifacts"),
        )


@dataclass(frozen=True)
class APIConfig:
    """Local HTTP API/UI settings."""

    host: str = "127.0.0.1"
    port: int = 8080
    #: Local-first: bind to loopback by default.  No stealth/proxy behaviour.
    allow_remote: bool = False
    static_dir: Path = field(default_factory=lambda: Path(__file__).parent / "ui" / "static")

    @classmethod
    def from_env(cls) -> "APIConfig":
        return cls(
            host=_env_str("MARKET_AI_HOST", "127.0.0.1") or "127.0.0.1",
            port=_env_int("MARKET_AI_PORT", 8080),
            allow_remote=_env_bool("MARKET_AI_ALLOW_REMOTE", False),
        )


@dataclass(frozen=True)
class Config:
    """Aggregate configuration object passed through the pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    api: APIConfig = field(default_factory=APIConfig)
    feature_version: str = FEATURE_VERSION
    strategy_version: str = STRATEGY_VERSION
    ensemble_version: str = ENSEMBLE_VERSION
    gate_version: str = GATE_VERSION
    regime_version: str = REGIME_VERSION
    seed: int = DEFAULT_SEED

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            data=DataConfig.from_env(),
            regime=RegimeConfig(),
            ensemble=EnsembleConfig.from_env(),
            decision=DecisionConfig.from_env(),
            risk=RiskConfig.from_env(),
            backtest=BacktestConfig.from_env(),
            promotion=PromotionConfig.from_env(),
            vision=VisionConfig.from_env(),
            reasoning=ReasoningConfig.from_env(),
            storage=StorageConfig.from_env(),
            api=APIConfig.from_env(),
            seed=_env_int("MARKET_AI_SEED", DEFAULT_SEED),
        )

    def with_overrides(self, **kwargs: Any) -> "Config":
        """Return a copy with top-level fields replaced (used by tests)."""
        return replace(self, **kwargs)

    def ensure_directories(self) -> None:
        """Create artifact directories.  Safe to call repeatedly."""
        for path in (
            self.storage.artifacts_dir,
            self.storage.db_path.parent,
            self.data.data_dir,
            self.vision.image_dir,
        ):
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except OSError:  # pragma: no cover - read-only filesystems
                pass

    def summary(self) -> Dict[str, Any]:
        """Compact, secret-free description used in reports and the UI."""
        return {
            "feature_version": self.feature_version,
            "strategy_version": self.strategy_version,
            "ensemble_version": self.ensemble_version,
            "gate_version": self.gate_version,
            "regime_version": self.regime_version,
            "seed": self.seed,
            "assets": list(self.data.assets),
            "timeframes": [t.value for t in self.data.timeframes],
            "vision_provider": self.vision.provider,
            "reasoning_provider": self.reasoning.provider,
            "payout": self.risk.payout,
            "breakeven_win_rate": round(self.risk.breakeven_win_rate, 4),
        }


def load_config(**overrides: Any) -> Config:
    """Build configuration from the environment plus explicit overrides."""
    cfg = Config.from_env()
    return cfg.with_overrides(**overrides) if overrides else cfg
