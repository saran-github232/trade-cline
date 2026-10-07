"""The end-to-end prediction pipeline.

This is the spine that implements the core system flow::

    MARKET DATA -> VALIDATION -> FEATURES -> TECHNICAL/PRICE-ACTION/STRUCTURE
      -> REGIME -> MULTI-TIMEFRAME -> ML -> VISION -> REASONING
      -> ENSEMBLE -> CALIBRATION -> DECISION GATE -> UP/DOWN/NO-TRADE
      -> PREDICTION LOG

Two invariants make the whole thing auditable:

1. **Causality.**  Every component receives ``history``, which is produced by
   ``MarketSeries.visible_at(as_of)``.  There is no code path that hands a
   component a candle closing after ``as_of``.
2. **Graceful degradation.**  Optional components (vision, reasoning, ML) are
   constructed lazily and, when missing or failing, contribute an
   ``available=False`` signal.  That *reduces* directional mass and therefore
   pushes towards NO-TRADE, which is the safe failure mode required by the
   stress-testing phase.  A failure never fabricates a direction.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import Config
from .ensemble.engine import EnsembleEngine, EnsembleResult
from .signals.base import SignalContext
from .signals.decision import DecisionGate, GateContext, GateResult
from .signals.providers import default_providers
from .types import (
    Candle,
    DataQuality,
    Decision,
    Direction,
    FeatureVector,
    MarketSeries,
    Prediction,
    ProbabilitySet,
    Regime,
    Signal,
    Timeframe,
)
from .utils.logging import get_logger, log_event
from .utils.stats import clamp, percentile, safe_div

__all__ = ["PredictionPipeline", "PipelineDiagnostics"]

_log = get_logger(__name__)


@dataclass
class PipelineDiagnostics:
    """Per-run health record.  Surfaced in the UI so failures are visible."""

    steps_ok: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    degraded_components: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def ok(self, name: str) -> None:
        self.steps_ok.append(name)

    def fail(self, name: str, exc: BaseException) -> None:
        self.steps_failed.append(f"{name}: {type(exc).__name__}: {exc}")
        self.degraded_components.append(name)

    @property
    def healthy(self) -> bool:
        return not self.steps_failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps_ok": list(self.steps_ok),
            "steps_failed": list(self.steps_failed),
            "degraded_components": list(self.degraded_components),
            "duration_ms": round(self.duration_ms, 3),
            "healthy": self.healthy,
        }


class PredictionPipeline:
    """Produces a fully-audited :class:`Prediction` at a point in time."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        model: Any = None,
        model_version: str = "untrained",
        dataset_version: str = "none",
        calibrator: Any = None,
        registry: Any = None,
        store: Any = None,
        db: Any = None,
        weights: Optional[Mapping[str, float]] = None,
        feature_engine: Any = None,
        regime_detector: Any = None,
        mtf_analyzer: Any = None,
        vision_analyser: Any = None,
        reasoning_supervisor: Any = None,
    ) -> None:
        self.config = config or Config()
        self.model = model
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.calibrator = calibrator
        self.registry = registry
        self.store = store
        self.db = db
        self.weights = dict(weights or self.config.ensemble.weights)

        self._feature_engine = feature_engine
        self._regime_detector = regime_detector
        self._mtf_analyzer = mtf_analyzer
        self._vision_analyser = vision_analyser
        self._reasoning_supervisor = reasoning_supervisor

        self.ensemble = EnsembleEngine(self.config, weights=self.weights)
        self.gate = DecisionGate(self.config)
        self.providers = default_providers(self.config.vision.max_influence)
        self.feature_names: List[str] = []

    # -- lazy component accessors -----------------------------------------
    # Components are built on first use and cached.  A missing optional
    # dependency must never prevent a NO-TRADE from being produced.
    @property
    def feature_engine(self):
        if self._feature_engine is None:
            from .features.feature_engine import FeatureEngine
            self._feature_engine = FeatureEngine(self.config)
            self.feature_names = list(self._feature_engine.feature_names())
        return self._feature_engine

    @property
    def regime_detector(self):
        if self._regime_detector is None:
            from .regime.detector import RegimeDetector
            self._regime_detector = RegimeDetector(self.config)
        return self._regime_detector

    @property
    def mtf_analyzer(self):
        if self._mtf_analyzer is None:
            from .multi_timeframe.analyzer import MultiTimeframeAnalyzer
            self._mtf_analyzer = MultiTimeframeAnalyzer(self.config)
        return self._mtf_analyzer

    @property
    def vision_analyser(self):
        if self._vision_analyser is None:
            from .vision.analyser import ChartVisionAnalyser
            self._vision_analyser = ChartVisionAnalyser(self.config)
        return self._vision_analyser

    @property
    def reasoning_supervisor(self):
        if self._reasoning_supervisor is None:
            from .reasoning.supervisor import ReasoningSupervisor
            self._reasoning_supervisor = ReasoningSupervisor(self.config)
        return self._reasoning_supervisor

    def set_model(self, model: Any, *, model_version: str, dataset_version: str = "unknown",
                  calibrator: Any = None) -> None:
        """Install the champion model (or any model) for subsequent predictions."""
        self.model = model
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.calibrator = calibrator

    # -- helper steps ------------------------------------------------------  
    def _compute_quality(self, series: MarketSeries) -> Tuple[DataQuality, float, List[str]]:
        """Validate the visible history; never raise on bad data."""
        try:
            from .data.validation import validate_series
            report = validate_series(series, self.config)
            return report.quality, report.quality.score, list(report.issues)
        except Exception as exc:  # pragma: no cover - defensive
            _log.debug("validation unavailable: %s", exc)
            return DataQuality.OK, 1.0, []

    def _ml_probability(self, features: FeatureVector, diag: PipelineDiagnostics) -> Optional[float]:
        """Model probability, or ``None`` when no usable model exists."""
        if self.model is None:
            return None
        try:
            if not self.feature_names:
                self.feature_names = list(self.feature_engine.feature_names())
            row = [features.get(name) for name in self.feature_names]
            probs = self.model.predict_proba([row])
            if not probs:
                return None
            value = float(probs[0])
            if value != value:  # NaN guard
                return None
            if self.calibrator is not None:
                try:
                    value = float(self.calibrator.transform([value])[0])
                except Exception as exc:
                    diag.fail("calibrator", exc)
            diag.ok("ml")
            return clamp(value)
        except Exception as exc:
            diag.fail("ml", exc)
            return None

    def _volatility_percentile(self, history: MarketSeries, features: FeatureVector) -> float:
        """Where current ATR sits relative to its own recent history.

        Used by the gate to refuse trades in volatility regimes the model was
        never validated on.
        """
        atr = features.get("volatility.atr_14")
        if atr != atr or len(history) < 40:
            return 0.5
        try:
            from .features.indicators import atr as atr_indicator
            series = atr_indicator(history.highs, history.lows, history.closes, 14)
            window = [v for v in series[-250:] if v == v]
            if len(window) < 20:
                return 0.5
            below = sum(1 for v in window if v <= atr)
            return clamp(safe_div(below, len(window), 0.5))
        except Exception:
            return 0.5

    # -- main entry point ---------------------------------------------------
    def predict(
        self,
        *,
        series_by_timeframe: Mapping[Timeframe, MarketSeries],
        asset: str,
        timeframe: Timeframe,
        as_of: int,
        horizon_bars: int = 3,
        historical_support: int = 0,
        record: bool = True,
        image_path: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Prediction:
        """Produce a decision for ``asset``/``timeframe`` as of ``as_of``.

        ``series_by_timeframe`` may contain higher and lower timeframes; each
        is truncated to what was knowable at ``as_of`` before use.
        """
        started = time.perf_counter()
        diag = PipelineDiagnostics()
        timeframe = Timeframe.parse(timeframe)
        tf_seconds = timeframe.seconds

        source = series_by_timeframe.get(timeframe)
        if source is None:
            raise ValueError(f"no series supplied for timeframe {timeframe.value}")

        # --- 1. CAUSALITY: truncate everything to as_of --------------------
        history = source.visible_at(as_of)
        if len(history) == 0:
            raise ValueError(f"no visible candles at {as_of} for {asset} {timeframe.value}")
        views = {
            tf: series.visible_at(as_of)
            for tf, series in series_by_timeframe.items()
        }

        quality, quality_score, quality_issues = self._compute_quality(history)
        diag.ok("data_validation")

        # --- 2. features ---------------------------------------------------
        try:
            features = self.feature_engine.compute(history)
            diag.ok("features")
        except Exception as exc:
            diag.fail("features", exc)
            features = FeatureVector(
                timestamp=as_of, asset=asset, timeframe=timeframe,
                version=self.config.feature_version, values={},
            )
        total_features = max(1, len(features.values))
        feature_coverage = clamp(safe_div(features.finite_count(), total_features, 0.0))

        # --- 3. regime -----------------------------------------------------
        regime_result = None
        try:
            regime_result = self.regime_detector.detect(history, features)
            diag.ok("regime")
        except Exception as exc:
            diag.fail("regime", exc)
        regime_label = Regime.UNCERTAIN
        regime_confidence = 0.0
        if regime_result is not None:
            regime_label = getattr(regime_result, "regime", Regime.UNCERTAIN)
            if not isinstance(regime_label, Regime):
                try:
                    regime_label = Regime(str(regime_label))
                except ValueError:
                    regime_label = Regime.UNCERTAIN
            regime_confidence = clamp(getattr(regime_result, "confidence", 0.0))

        # --- 4. multi-timeframe --------------------------------------------
        mtf_result = None
        if len(views) > 1:
            try:
                mtf_result = self.mtf_analyzer.analyse(views, as_of, timeframe)
                diag.ok("multi_timeframe")
            except Exception as exc:
                diag.fail("multi_timeframe", exc)

        # --- 5. ML ---------------------------------------------------------
        ml_probability = self._ml_probability(features, diag)

        # --- 6. vision -----------------------------------------------------
        vision_result = None
        try:
            summary = self._market_summary(history, features, regime_label, regime_confidence)
            vision_result = self.vision_analyser.analyse(
                series=history, summary=summary, image_path=image_path,
                timeframe=timeframe, asset=asset,
            )
            diag.ok("vision")
        except Exception as exc:
            diag.fail("vision", exc)

        # --- 7. signals ----------------------------------------------------
        context = SignalContext(
            asset=asset, timeframe=timeframe, as_of=as_of, series=history,
            features=features, regime=regime_result, mtf=mtf_result,
            ml_probability=ml_probability, vision=vision_result,
            quality_score=quality_score, extra=dict(extra or {}),
        )
        signals: List[Signal] = []
        for provider in self.providers:
            try:
                signals.append(provider.compute(context))
            except Exception as exc:
                diag.fail(f"signal:{provider.name}", exc)
                from .signals.base import unavailable_signal
                signals.append(unavailable_signal(provider.name, as_of, str(exc)))

        # --- 8. reasoning supervisor ---------------------------------------
        reasoning_result = None
        try:
            reasoning_result = self.reasoning_supervisor.review(
                context=self._reasoning_context(
                    asset, timeframe, as_of, features, regime_label,
                    regime_confidence, ml_probability, vision_result,
                    mtf_result, signals, quality_score, historical_support,
                )
            )
            diag.ok("reasoning")
        except Exception as exc:
            diag.fail("reasoning", exc)

        # --- 9. ensemble ---------------------------------------------------
        ensemble_result = self.ensemble.combine(
            signals, timestamp=as_of, context=context, reasoning=reasoning_result,
        )
        diag.ok("ensemble")

        # --- 10. confidence ------------------------------------------------
        confidence = self._confidence(ensemble_result, quality_score, feature_coverage)

        # --- 11. gate ------------------------------------------------------
        volatility_pct = self._volatility_percentile(history, features)
        gate_result = self.gate.decide(
            GateContext(
                probabilities=ensemble_result.probabilities,
                confidence=confidence,
                agreement=ensemble_result.agreement,
                timestamp=as_of,
                regime=regime_label,
                regime_confidence=regime_confidence,
                quality=quality,
                feature_coverage=feature_coverage,
                model_available=ml_probability is not None,
                model_degraded=self.model is None,
                historical_support=historical_support,
                volatility_percentile=volatility_pct,
                out_of_distribution=self._is_out_of_distribution(features, history),
                signals=signals,
            )
        )
        diag.ok("gate")

        diag.duration_ms = (time.perf_counter() - started) * 1000.0

        prediction = self._build_prediction(
            asset=asset, timeframe=timeframe, as_of=as_of, history=history,
            features=features, regime=regime_label, signals=signals,
            ensemble_result=ensemble_result, gate_result=gate_result,
            confidence=confidence, horizon_bars=horizon_bars,
            vision_result=vision_result, reasoning_result=reasoning_result,
            diag=diag, quality=quality, quality_issues=quality_issues,
            volatility_percentile=volatility_pct,
        )

        if record:
            self._record(prediction)
        return prediction

    # -- supporting builders ------------------------------------------------
    def _market_summary(
        self, history: MarketSeries, features: FeatureVector,
        regime: Regime, regime_confidence: float,
    ) -> Dict[str, Any]:
        """Structured numerical summary handed to the vision component.

        The vision model is told exactly what the numbers say.  This is what
        stops it from inventing invisible levels: it is asked to interpret a
        chart *against* a known numerical context, not from scratch.
        """
        closes = history.closes
        last = history[-1]
        window = closes[-60:] if len(closes) >= 60 else closes
        return {
            "asset": history.asset,
            "timeframe": history.timeframe.value,
            "as_of": last.close_time,
            "price": last.close,
            "bars": len(history),
            "range_high": max(history.highs[-60:]) if len(history) >= 1 else last.high,
            "range_low": min(history.lows[-60:]) if len(history) >= 1 else last.low,
            "change_pct": round(safe_div(last.close - window[0], window[0], 0.0) * 100.0, 4) if window else 0.0,
            "regime": regime.value,
            "regime_confidence": regime_confidence,
            "features": {k: v for k, v in features.values.items() if v == v},
        }

    def _reasoning_context(
        self, asset: str, timeframe: Timeframe, as_of: int, features: FeatureVector,
        regime: Regime, regime_confidence: float, ml_probability: Optional[float],
        vision: Any, mtf: Any, signals: Sequence[Signal], quality_score: float,
        historical_support: int,
    ) -> Dict[str, Any]:
        """Assemble the evidence pack the supervisor reasons over."""
        return {
            "asset": asset,
            "timeframe": timeframe.value,
            "as_of": as_of,
            "regime": regime.value,
            "regime_confidence": regime_confidence,
            "ml_probability": ml_probability,
            "vision": vision.to_dict() if hasattr(vision, "to_dict") else None,
            "multi_timeframe": mtf.to_dict() if hasattr(mtf, "to_dict") else None,
            "signals": [s.to_dict() for s in signals],
            "quality_score": quality_score,
            "historical_support": historical_support,
            "key_features": {
                k: round(v, 6) for k, v in features.values.items()
                if v == v and k in (
                    "momentum.rsi_14", "trend.ema_9", "trend.ema_21", "trend.ema_50",
                    "volatility.atr_14", "volatility.atr_percentile",
                    "structure.breakout_up", "structure.breakout_down",
                    "pa.body_ratio", "momentum.adx_14",
                )
            },
        }

    def _confidence(
        self, ensemble: EnsembleResult, quality_score: float, feature_coverage: float
    ) -> float:
        """Confidence in the *decision*, not in the direction.

        It combines how much directional mass the evidence earned, how much
        the providers agreed, and how trustworthy the inputs were.  A high
        P(UP) with terrible data must NOT produce high confidence.
        """
        mass = clamp(ensemble.directional_mass)
        agreement = clamp(ensemble.agreement)
        edge = clamp(abs(ensemble.mean_probability - 0.5) * 2.0)
        raw = 0.45 * mass + 0.30 * agreement + 0.25 * edge
        return clamp(raw * clamp(quality_score) * (0.6 + 0.4 * clamp(feature_coverage)))

    @staticmethod
    def _is_out_of_distribution(features: FeatureVector, history: MarketSeries) -> bool:
        """Cheap OOD screen on the few features with a known sane range.

        This is deliberately conservative and explainable rather than a
        learned density model: an unexplained NO-TRADE is acceptable, a silent
        extrapolation is not.
        """
        rsi = features.get("momentum.rsi_14")
        if rsi == rsi and (rsi < 0.0 or rsi > 100.0):
            return True
        pct_b = features.get("volatility.bb_pct_b")
        if pct_b == pct_b and abs(pct_b) > 6.0:
            return True
        if len(history) < 30:
            return True
        closes = history.closes[-30:]
        if any(c <= 0 for c in closes):
            return True
        return False

    def _build_prediction(
        self, *, asset: str, timeframe: Timeframe, as_of: int, history: MarketSeries,
        features: FeatureVector, regime: Regime, signals: Sequence[Signal],
        ensemble_result: EnsembleResult, gate_result: GateResult, confidence: float,
        horizon_bars: int, vision_result: Any, reasoning_result: Any,
        diag: PipelineDiagnostics, quality: DataQuality, quality_issues: Sequence[str],
        volatility_percentile: float,
    ) -> Prediction:
        """Assemble the immutable, fully-audited prediction record."""
        entry = history[-1].close
        horizon_seconds = max(1, horizon_bars) * timeframe.seconds
        expires_at = history[-1].close_time + horizon_seconds
        decision = gate_result.decision

        # Confidence reported to the user is gated by the actual decision: a
        # NO-TRADE is reported with the confidence of *abstaining*, which is
        # high when evidence is genuinely absent.
        reported_confidence = confidence if decision is not Decision.NO_TRADE else max(
            confidence, ensemble_result.probabilities.p_no_trade * 0.5 + 0.5 * confidence
        )

        return Prediction(
            prediction_id=f"{asset}-{timeframe.value}-{as_of}-{uuid.uuid4().hex[:8]}",
            timestamp=as_of,
            asset=asset,
            timeframe=timeframe,
            decision=decision,
            probabilities=ensemble_result.probabilities,
            confidence=clamp(reported_confidence),
            regime=regime,
            agreement=ensemble_result.agreement,
            reason=self.gate.explain(gate_result),
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            feature_version=getattr(features, "version", self.config.feature_version),
            strategy_version=self.config.strategy_version,
            horizon_seconds=horizon_seconds,
            entry_price=entry,
            expires_at=expires_at,
            signals=tuple(signals),
            features={k: v for k, v in features.values.items() if v == v},
            vision=vision_result.to_dict() if hasattr(vision_result, "to_dict") else {},
            reasoning=reasoning_result.to_dict() if hasattr(reasoning_result, "to_dict") else {},
            diagnostics={
                **diag.to_dict(),
                "gate": gate_result.to_dict(),
                "ensemble": {
                    "directional_mass": round(ensemble_result.directional_mass, 6),
                    "disagreement": round(ensemble_result.disagreement, 6),
                    "weights_used": {k: round(v, 6) for k, v in ensemble_result.weights_used.items()},
                    "modifiers": {k: round(v, 6) for k, v in ensemble_result.modifiers.items()},
                    "notes": list(ensemble_result.notes),
                },
                "data_quality": {"quality": quality.value, "issues": list(quality_issues)},
                "volatility_percentile": round(volatility_percentile, 4),
            },
        )

    def _record(self, prediction: Prediction) -> None:
        """Persist to the prediction log.  Logging failure must not lose a prediction."""
        try:
            if self.db is not None:
                self.db.predictions.add(prediction)
        except Exception as exc:
            log_event(_log, 30, "failed to persist prediction", error=str(exc))
