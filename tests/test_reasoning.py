"""Tests for the reasoning supervisor (section 8 of INTERFACES.md)."""

import unittest

from market_ai.config import Config
from market_ai.reasoning.schema import validate_reasoning_payload
from market_ai.reasoning.supervisor import ReasoningSupervisor
from market_ai.types import Decision


class _BadProvider:
    """Returns a structurally malformed payload."""

    name = "bad"

    def review(self, *, context):
        return {"foo": "bar"}


class _RaisingProvider:
    """Always raises."""

    name = "raising"

    def review(self, *, context):
        raise RuntimeError("boom")


class _UpgraderProvider:
    """Tries to upgrade the decision - the supervisor must ignore this."""

    name = "upgrader"

    def review(self, *, context):
        return {
            "recommendation": "UP",
            "contradictions": [],
            "weak_evidence": [],
            "invalidating_conditions": [],
            "abnormal_conditions": [],
            "explanation": "looks bullish",
            "confidence_adjustment": 0.0,
        }


class _NoTradeProvider:
    """Returns a valid payload recommending NO-TRADE."""

    name = "notrade"

    def review(self, *, context):
        return {
            "recommendation": "NO-TRADE",
            "contradictions": ["vision disagrees"],
            "weak_evidence": [],
            "invalidating_conditions": [],
            "abnormal_conditions": [],
            "explanation": "conflicting evidence",
            "confidence_adjustment": -0.2,
        }


def _contradictory_context():
    return {
        "decision": "UP",
        "technical": {"direction": "UP", "confidence": 0.9},
        "ml": {"direction": "DOWN", "confidence": 0.9},
        "vision": {"direction": "DOWN", "confidence": 0.8},
        "regime": {"direction": "DOWN", "confidence": 0.7},
    }


def _aligned_context():
    return {
        "decision": "UP",
        "technical": {"direction": "UP", "confidence": 0.8},
        "ml": {"direction": "UP", "confidence": 0.8},
        "vision": {"direction": "UP", "confidence": 0.7},
        "regime": {"direction": "UP", "confidence": 0.7},
    }


class ReasoningSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = ReasoningSupervisor(Config())

    def test_contradictory_evidence_produces_no_trade(self):
        result = self.supervisor.review(context=_contradictory_context())
        self.assertEqual(result.recommendation, Decision.NO_TRADE)
        self.assertTrue(result.contradictions)
        self.assertLessEqual(result.confidence_adjustment, 0.0)
        self.assertTrue(result.explanation)

    def test_aligned_evidence_does_not_upgrade(self):
        result = self.supervisor.review(context=_aligned_context())
        self.assertEqual(result.recommendation, Decision.UP)
        self.assertEqual(result.contradictions, [])
        self.assertLessEqual(result.confidence_adjustment, 0.0)

    def test_confidence_adjustment_is_never_positive(self):
        contexts = [
            _contradictory_context(),
            _aligned_context(),
            {
                "decision": "DOWN",
                "technical": {"direction": "DOWN", "confidence": 0.5},
                "feature_coverage": 0.3,
                "volatility": 5.0,
                "volatility_band": [0.5, 1.5],
                "data_quality": "BAD",
            },
            {"decision": "UP"},
        ]
        for context in contexts:
            result = self.supervisor.review(context=context)
            self.assertLessEqual(result.confidence_adjustment, 0.0)
            self.assertGreaterEqual(result.confidence_adjustment, -1.0)

    def test_abnormal_conditions_are_detected(self):
        context = {
            "decision": "UP",
            "technical": {"direction": "UP", "confidence": 0.7},
            "feature_coverage": 0.4,
            "volatility": 4.0,
            "volatility_band": [0.5, 1.5],
            "vision_degraded": True,
            "data_quality": "DEGRADED",
        }
        result = self.supervisor.review(context=context)
        self.assertTrue(result.abnormal_conditions)
        self.assertEqual(result.recommendation, Decision.NO_TRADE)  # severe coverage

    def test_missing_base_decision_defaults_to_no_trade(self):
        result = self.supervisor.review(context={"technical": {"direction": "UP"}})
        self.assertEqual(result.recommendation, Decision.NO_TRADE)

    def test_weak_evidence_is_flagged(self):
        context = {
            "decision": "UP",
            "technical": {"direction": "UP", "confidence": 0.2},
            "ml": {"direction": "UP", "confidence": 0.9, "available": False},
        }
        result = self.supervisor.review(context=context)
        self.assertTrue(result.weak_evidence)

    def test_malformed_provider_output_degrades_safely(self):
        supervisor = ReasoningSupervisor(Config(), provider=_BadProvider())
        result = supervisor.review(context=_contradictory_context())
        self.assertTrue(result.degraded)
        self.assertEqual(result.provider, "offline")
        self.assertLessEqual(result.confidence_adjustment, 0.0)
        self.assertIn(result.recommendation, (Decision.UP, Decision.NO_TRADE))

    def test_raising_provider_degrades_safely(self):
        supervisor = ReasoningSupervisor(Config(), provider=_RaisingProvider())
        result = supervisor.review(context=_aligned_context())
        self.assertTrue(result.degraded)
        self.assertLessEqual(result.confidence_adjustment, 0.0)

    def test_provider_cannot_upgrade_when_base_is_no_trade(self):
        supervisor = ReasoningSupervisor(Config(), provider=_UpgraderProvider())
        result = supervisor.review(context={"decision": "NO-TRADE"})
        self.assertEqual(result.recommendation, Decision.NO_TRADE)
        self.assertFalse(result.degraded)

    def test_provider_cannot_reverse_direction(self):
        supervisor = ReasoningSupervisor(Config(), provider=_UpgraderProvider())
        result = supervisor.review(context={"decision": "DOWN"})
        self.assertEqual(result.recommendation, Decision.DOWN)

    def test_valid_provider_can_push_towards_no_trade(self):
        supervisor = ReasoningSupervisor(Config(), provider=_NoTradeProvider())
        result = supervisor.review(context={"decision": "UP"})
        self.assertEqual(result.recommendation, Decision.NO_TRADE)
        self.assertFalse(result.degraded)
        self.assertLessEqual(result.confidence_adjustment, 0.0)

    def test_validate_reasoning_payload_rejects_positive_adjustment(self):
        payload = {
            "recommendation": "UP",
            "contradictions": [],
            "weak_evidence": [],
            "invalidating_conditions": [],
            "abnormal_conditions": [],
            "explanation": "",
            "confidence_adjustment": 0.3,
        }
        ok, errors = validate_reasoning_payload(payload)
        self.assertFalse(ok)
        self.assertTrue(any("confidence_adjustment" in e for e in errors))


if __name__ == "__main__":
    unittest.main()

