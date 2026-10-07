"""Application service container.

One object that owns every long-lived component (database, registry,
experience store, pipeline, backtester, paper-trading engine) and exposes the
few operations the API, the CLI and the tests actually need.

Everything here is defensive: a missing optional component degrades the
feature that needs it, but never prevents the system from producing a
NO-TRADE, which is the safe answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import Config, load_config
from .experience.store import ExperienceStore
from .outcomes.resolver import OutcomeResolver
from .pipeline import PredictionPipeline
from .storage.db import Database
from .types import Decision, Direction, MarketSeries, Prediction, Timeframe
from .utils.logging import get_logger, log_event

__all__ = ["Application", "build_demo_application"]

_log = get_logger(__name__)


@dataclass
class AppState:
    """Lightweight snapshot used by the dashboard."""

    ready: bool
    champion: Optional[str]
    counts: Dict[str, int]
    degraded: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "champion": self.champion,
            "counts": self.counts,
            "degraded": self.degraded,
        }


class Application:
    """Composition root for the whole system."""

    def __init__(self, config: Optional[Config] = None, *, use_db: bool = True) -> None:
        self.config = config or load_config()
        self.config.ensure_directories()
        self.degraded: List[str] = []

        self.db: Optional[Database] = None
        if use_db:
            try:
                self.db = Database(self.config.storage.db_path).initialize()
            except Exception as exc:  # pragma: no cover - filesystem issues
                self.degraded.append(f"database: {exc}")
                self.db = None

        self.registry = self._build_registry()
        self.store = ExperienceStore(self.db, self.config)
        self.outcome_resolver = OutcomeResolver(self.config)
        self.pipeline = PredictionPipeline(self.config, db=self.db, registry=self.registry)
        self._champion_model: Any = None
        self._champion_version = "untrained"
        self._champion_dataset = "none"
        self._calibrator: Any = None

        self._provider = self._build_provider()
        self.load_champion()

    # -- construction helpers ---------------------------------------------
    def _build_registry(self):
        try:
            from .models.registry import ModelRegistry
            return ModelRegistry(self.config.storage.artifacts_dir / "registry")
        except Exception as exc:  # pragma: no cover
            self.degraded.append(f"registry: {exc}")
            return None

    def _build_provider(self):
        """Build a local data provider (no network, by design)."""
        try:
            from .data.providers import HistoricalDataProvider
            from .data.synthetic import SyntheticProvider

            provider = HistoricalDataProvider({})
            synthetic = SyntheticProvider(self.config)
            for asset in self.config.data.assets:
                for timeframe in self.config.data.timeframes:
                    try:
                        series = synthetic.get_series(asset, timeframe)
                        provider.add_series(series)
                    except Exception as exc:
                        self.degraded.append(f"synthetic:{asset}:{timeframe.value}: {exc}")
            return provider
        except Exception as exc:
            self.degraded.append(f"data_provider: {exc}")
            return None

    # -- champion lifecycle ------------------------------------------------
    def load_champion(self) -> Optional[str]:
        """Load the current champion from the registry into the pipeline."""
        if self.registry is None:
            return None
        try:
            record = self.registry.champion()
            if record is None:
                log_event(_log, 20, "no champion registered; pipeline will emit NO-TRADE")
                return None
            model = self.registry.load(record.model_version)
            calibrator = None
            cal_payload = (record.calibration or {}).get("calibrator")
            if cal_payload:
                try:
                    from .models.calibration import Calibrator
                    calibrator = Calibrator.from_dict(cal_payload)
                except Exception as exc:
                    self.degraded.append(f"calibrator: {exc}")
            self._champion_model = model
            self._champion_version = record.model_version
            self._champion_dataset = record.dataset_version
            self._calibrator = calibrator
            self.pipeline.set_model(
                model, model_version=record.model_version,
                dataset_version=record.dataset_version, calibrator=calibrator,
            )
            return record.model_version
        except Exception as exc:
            self.degraded.append(f"load_champion: {exc}")
            return None

    @property
    def champion_version(self) -> str:
        return self._champion_version

    # -- market data -------------------------------------------------------
    def series(self, asset: str, timeframe: Timeframe) -> Optional[MarketSeries]:
        if self._provider is None:
            return None
        try:
            return self._provider.get_series(asset, Timeframe.parse(timeframe))
        except Exception as exc:
            self.degraded.append(f"series:{asset}:{timeframe}: {exc}")
            return None

    def series_bundle(self, asset: str, timeframe: Timeframe) -> Dict[Timeframe, MarketSeries]:
        """All available timeframes for an asset, for multi-timeframe analysis."""
        bundle: Dict[Timeframe, MarketSeries] = {}
        for tf in self.config.data.timeframes:
            series = self.series(asset, tf)
            if series is not None and len(series):
                bundle[tf] = series
        if timeframe not in bundle:
            single = self.series(asset, timeframe)
            if single is not None:
                bundle[timeframe] = single
        return bundle

    # -- prediction --------------------------------------------------------
    def predict(
        self, asset: str, timeframe: Timeframe, *, as_of: Optional[int] = None,
        horizon_bars: int = 3, record: bool = True,
    ) -> Optional[Prediction]:
        """Produce a prediction at the latest (or a specified) closed bar."""
        timeframe = Timeframe.parse(timeframe)
        bundle = self.series_bundle(asset, timeframe)
        if timeframe not in bundle:
            return None
        series = bundle[timeframe]
        if as_of is None:
            as_of = series[-1].close_time
        support = self._historical_support(asset, timeframe)
        try:
            return self.pipeline.predict(
                series_by_timeframe=bundle, asset=asset, timeframe=timeframe,
                as_of=as_of, horizon_bars=horizon_bars,
                historical_support=support, record=record,
            )
        except Exception as exc:
            log_event(_log, 40, "prediction failed", asset=asset,
                      timeframe=timeframe.value, error=str(exc))
            return None

    def _historical_support(self, asset: str, timeframe: Timeframe) -> int:
        """How many resolved experiences exist for this asset/timeframe."""
        if self.db is None:
            return 0
        try:
            return len([
                e for e in self.db.experiences.list(order_by="timestamp")
                if e.get("asset") == asset and e.get("timeframe") == timeframe.value
            ])
        except Exception:
            return 0

    # -- outcome resolution and learning -----------------------------------
    def resolve_outcomes(self, *, limit: int = 500) -> List[Any]:
        """Match logged predictions with realised outcomes.

        This is the automatic-learning entry point: it is safe to call on a
        schedule, and it is idempotent because the experience table is keyed by
        prediction id.
        """
        if self.db is None:
            return []
        from .experience.store import build_experience

        resolved: List[Any] = []
        try:
            pending = self.db.predictions.unresolved(limit=limit)
        except Exception as exc:
            self.degraded.append(f"resolve_outcomes: {exc}")
            return []

        for payload in pending:
            try:
                prediction = Prediction.from_dict(payload)
            except (KeyError, TypeError, ValueError):
                self.db.predictions.mark_resolved(payload.get("prediction_id", ""))
                continue
            series = self.series(prediction.asset, prediction.timeframe)
            if series is None:
                continue
            outcome = self.outcome_resolver.resolve(prediction, series)
            if outcome is None:
                continue  # horizon has not elapsed yet
            try:
                self.db.outcomes.add(outcome)
                self.db.predictions.mark_resolved(prediction.prediction_id)
                experience = build_experience(prediction, outcome)
                self.db.experiences.add(experience)
                resolved.append(experience)
            except Exception as exc:
                self.degraded.append(f"store_outcome:{prediction.prediction_id}: {exc}")
        return resolved

    # -- dashboard state ---------------------------------------------------
    def state(self) -> AppState:
        counts: Dict[str, int] = {}
        if self.db is not None:
            try:
                counts = self.db.stats()
            except Exception as exc:
                self.degraded.append(f"stats: {exc}")
        champion = None
        if self.registry is not None:
            try:
                record = self.registry.champion()
                champion = record.model_version if record else None
            except Exception:
                champion = None
        return AppState(
            ready=self.pipeline is not None,
            champion=champion,
            counts=counts,
            degraded=sorted(set(self.degraded)),
        )

    def close(self) -> None:
        if self.db is not None:
            self.db.close()


def build_demo_application(config: Optional[Config] = None) -> Application:
    """Build an application wired for offline demo/testing use."""
    return Application(config or load_config(), use_db=True)
