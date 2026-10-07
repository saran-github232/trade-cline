"""Tests for the ensemble engine and the decision gate.

These are the tests that matter most for the headline requirement: the system
must be ABLE and WILLING to say NO-TRADE.
"""

import unittest

from market_ai.config import Config
from market_ai.ensemble.engine import EnsembleEngine, agreement_score, disagreement_score
from market_ai.signals.decision import DecisionGate, GateContext
from market_ai.types import (
    DataQuality,
    Decision,
    Direction,
    ProbabilitySet,
    Regime,
    Signal,
    Timeframe,
)


def sig(source, direction, p, conf):
    return Signal(source, direction, p, conf, 0)


ALIGNED_UP = [
    sig("technical", Direction.UP, 0.78, 0.85),
    sig("price_action", Direction.UP, 0.74, 0.80),
    sig("structure", Direction.UP, 0.72, 0.75),
    sig("regime", Direction.UP, 0.70, 0.80),
    sig("ml", Direction.UP, 0.76, 0.65),
    sig("vision", Direction.UP, 0.65, 0.30),
]

CONFLICTED = [
    sig("technical", Direction.UP, 0.90, 0.90),
    sig("price_action", Direction.DOWN, 0.10, 0.90),
    sig("structure", Direction.NEUTRAL, 0.50, 0.50),
    sig("regime", Direction.NEUTRAL, 0.50, 0.10),
    sig("ml", Direction.UP, 0.85, 0.80),
    sig("vision", Direction.DOWN, 0.20, 0.60),
]


class AgreementTests(unittest.TestCase):
    def test_perfect_agreement(self):
        same = [sig("a", Direction.UP, 0.8, 1.0), sig("b", Direction.UP, 0.8, 1.0)]
        self.assertAlmostEqual(disagreement_score(same), 0.0)
        self.assertGreater(agreement_score(same), 0.9)

    def test_maximal_disagreement(self):
        split = [sig("a", Direction.UP, 1.0, 1.0), sig("b", Direction.DOWN, 0.0, 1.0)]
        self.assertAlmostEqual(disagreement_score(split), 1.0, places=6)

    def test_single_provider_has_no_disagreement(self):
        self.assertEqual(disagreement_score([sig("a", Direction.UP, 0.9, 1.0)]), 0.0)

    def test_unconfident_providers_do_not_create_disagreement(self):
        noisy = [sig("a", Direction.UP, 0.9, 0.01), sig("b", Direction.DOWN, 0.1, 0.01)]
        self.assertLess(disagreement_score(noisy), 0.5)


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.engine = EnsembleEngine(Config())

    def test_probabilities_normalised(self):
        r = self.engine.combine(ALIGNED_UP, timestamp=0)
        p = r.probabilities
        self.assertAlmostEqual(p.p_up + p.p_down + p.p_no_trade, 1.0, places=9)

    def test_aligned_evidence_produces_directional_mass(self):
        r = self.engine.combine(ALIGNED_UP, timestamp=0)
        self.assertGreater(r.probabilities.p_up, r.probabilities.p_down)
        self.assertGreater(r.directional_mass, 0.0)

    def test_conflict_collapses_directional_mass(self):
        aligned = self.engine.combine(ALIGNED_UP, timestamp=0)
        conflicted = self.engine.combine(CONFLICTED, timestamp=0)
        self.assertLess(conflicted.directional_mass, aligned.directional_mass)
        self.assertGreater(conflicted.probabilities.p_no_trade, aligned.probabilities.p_no_trade)

    def test_conflict_prefers_no_trade(self):
        r = self.engine.combine(CONFLICTED, timestamp=0)
        self.assertIs(r.decision, Decision.NO_TRADE)

    def test_no_signals_is_unambiguous_no_trade(self):
        r = self.engine.combine([], timestamp=0)
        self.assertEqual(r.probabilities.p_no_trade, 1.0)
        self.assertEqual(r.directional_mass, 0.0)
        self.assertIs(r.decision, Decision.NO_TRADE)

    def test_unavailable_providers_reduce_mass(self):
        full = self.engine.combine(ALIGNED_UP, timestamp=0)
        partial = self.engine.combine(ALIGNED_UP[:2], timestamp=0)
        self.assertLess(partial.directional_mass, full.directional_mass)

    def test_mass_is_monotonic_in_confidence(self):
        weak = [sig("technical", Direction.UP, 0.8, 0.2), sig("ml", Direction.UP, 0.8, 0.2)]
        strong = [sig("technical", Direction.UP, 0.8, 0.9), sig("ml", Direction.UP, 0.8, 0.9)]
        self.assertLess(
            self.engine.combine(weak, timestamp=0).directional_mass,
            self.engine.combine(strong, timestamp=0).directional_mass,
        )

    def test_reasoning_can_only_reduce_mass(self):
        class FakeReasoning:
            contradictions = ["ml disagrees with structure"]
            confidence_adjustment = -0.3

        plain = self.engine.combine(ALIGNED_UP, timestamp=0)
        reviewed = self.engine.combine(ALIGNED_UP, timestamp=0, reasoning=FakeReasoning())
        self.assertLessEqual(reviewed.directional_mass, plain.directional_mass)

    def test_positive_reasoning_adjustment_is_ignored(self):
        class UpwardReasoning:
            contradictions = []
            confidence_adjustment = 0.9  # must be clamped to 0 by the engine

        plain = self.engine.combine(ALIGNED_UP, timestamp=0)
        reviewed = self.engine.combine(ALIGNED_UP, timestamp=0, reasoning=UpwardReasoning())
        self.assertLessEqual(reviewed.directional_mass, plain.directional_mass + 1e-12)

    def test_weights_are_configurable(self):
        engine = EnsembleEngine(Config(), weights={"technical": 1.0, "ml": 0.0})
        r = engine.combine(ALIGNED_UP, timestamp=0)
        self.assertGreater(r.weights_used["technical"], 0.0)
        self.assertEqual(r.weights_used["ml"], 0.0)

    def test_serialisation(self):
        r = self.engine.combine(ALIGNED_UP, timestamp=0)
        d = r.to_dict()
        self.assertIn("probabilities", d)
        self.assertEqual(len(d["signals"]), len(ALIGNED_UP))


class DecisionGateTests(unittest.TestCase):
    """The gate is the last line of defence; every veto path must be tested."""

    def setUp(self):
        self.gate = DecisionGate(Config())

    def _ctx(self, **overrides):
        base = dict(
            probabilities=ProbabilitySet(0.55, 0.15, 0.30),
            confidence=0.75,
            agreement=0.80,
            timestamp=0,
            regime=Regime.STRONG_UPTREND,
            regime_confidence=0.80,
            quality=DataQuality.OK,
            feature_coverage=1.0,
            model_available=True,
            model_degraded=False,
            historical_support=500,
            volatility_percentile=0.5,
            out_of_distribution=False,
        )
        base.update(overrides)
        return GateContext(**base)

    def test_happy_path_produces_up(self):
        r = self.gate.decide(self._ctx())
        self.assertIs(r.decision, Decision.UP)
        self.assertEqual(r.vetoes, [])

    def test_happy_path_produces_down(self):
        r = self.gate.decide(self._ctx(probabilities=ProbabilitySet(0.15, 0.55, 0.30)))
        self.assertIs(r.decision, Decision.DOWN)

    def test_bad_data_forces_no_trade(self):
        r = self.gate.decide(self._ctx(quality=DataQuality.BAD))
        self.assertIs(r.decision, Decision.NO_TRADE)
        self.assertTrue(any("DATA_BAD" in v for v in r.vetoes))

    def test_model_failure_forces_no_trade(self):
        r = self.gate.decide(self._ctx(model_available=False))
        self.assertIs(r.decision, Decision.NO_TRADE)
        self.assertTrue(any("MODEL_UNAVAILABLE" in v for v in r.vetoes))

    def test_degraded_model_forces_no_trade(self):
        r = self.gate.decide(self._ctx(model_degraded=True))
        self.assertIs(r.decision, Decision.NO_TRADE)

    def test_unknown_regime_forces_no_trade(self):
        r = self.gate.decide(self._ctx(regime=Regime.UNCERTAIN, regime_confidence=0.0))
        self.assertIs(r.decision, Decision.NO_TRADE)
        self.assertTrue(any("REGIME_UNKNOWN" in v for v in r.vetoes))

    def test_low_regime_confidence_forces_no_trade(self):
        r = self.gate.decide(self._ctx(regime=Regime.RANGE, regime_confidence=0.05))
        self.assertIs(r.decision, Decision.NO_TRADE)

    def test_conflicting_signals_force_no_trade(self):
        r = self.gate.decide(self._ctx(agreement=0.20))
        self.assertIs(r.decision, Decision.NO_TRADE)
        self.assertTrue(any("SIGNALS_CONFLICT" in v for v in r.vetoes))

    def test_low_confidence_forces_no_trade(self):
        r = self.gate.decide(self._ctx(confidence=0.30))
        self.assertIs(r.decision, Decision.NO_TRADE)

    def test_incomplete_features_force_no_trade(self):
        r = self.gate.decide(self._ctx(feature_coverage=0.30))
        self.assertIs(r.decision, Decision.NO_TRADE)

    def test_insufficient_history_forces_no_trade(self):
        r = self.gate.decide(self._ctx(historical_support=3))
        self.assertIs(r.decision, Decision.NO_TRADE)

    def test_out_of_distribution_forces_no_trade(self):
        r = self.gate.decide(self._ctx(out_of_distribution=True))
        self.assertIs(r.decision, Decision.NO_TRADE)
        self.assertTrue(any("OUT_OF_DISTRIBUTION" in v for v in r.vetoes))

    def test_extreme_volatility_forces_no_trade(self):
        r = self.gate.decide(self._ctx(volatility_percentile=0.999))
        self.assertIs(r.decision, Decision.NO_TRADE)
        self.assertTrue(any("VOLATILITY_OUT_OF_RANGE" in v for v in r.vetoes))

    def test_narrow_margin_forces_no_trade(self):
        r = self.gate.decide(self._ctx(probabilities=ProbabilitySet(0.34, 0.33, 0.33)))
        self.assertIs(r.decision, Decision.NO_TRADE)

    def test_no_trade_dominant_probability_forces_no_trade(self):
        r = self.gate.decide(self._ctx(probabilities=ProbabilitySet(0.30, 0.25, 0.45)))
        self.assertIs(r.decision, Decision.NO_TRADE)

    def test_every_veto_is_explained(self):
        r = self.gate.decide(self._ctx(
            quality=DataQuality.BAD, model_available=False,
            regime=Regime.UNCERTAIN, agreement=0.1, confidence=0.1,
            feature_coverage=0.1, historical_support=0,
            out_of_distribution=True, volatility_percentile=1.0,
        ))
        self.assertIs(r.decision, Decision.NO_TRADE)
        # Uncertainty must never be hidden: every reason is reported.
        self.assertGreaterEqual(len(r.vetoes), 8)
        self.assertTrue(r.reasons)

    def test_explain_is_human_readable(self):
        ok = self.gate.decide(self._ctx())
        self.assertIn("UP", self.gate.explain(ok))
        veto = self.gate.decide(self._ctx(quality=DataQuality.BAD))
        self.assertIn("NO-TRADE", self.gate.explain(veto))

    def test_gate_never_returns_invalid_decision(self):
        for prob in (ProbabilitySet(0.9, 0.05, 0.05), ProbabilitySet(0.05, 0.9, 0.05),
                     ProbabilitySet(0.33, 0.33, 0.34)):
            r = self.gate.decide(self._ctx(probabilities=prob))
            self.assertIn(r.decision, (Decision.UP, Decision.DOWN, Decision.NO_TRADE))


if __name__ == "__main__":
    unittest.main()
