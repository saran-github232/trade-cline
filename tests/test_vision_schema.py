"""Tests for market_ai.vision.schema."""

import unittest

from market_ai.types import Direction
from market_ai.vision.schema import (
    VISION_RESPONSE_SCHEMA,
    VisionResult,
    coerce_payload,
    validate_vision_payload,
)


def _valid_payload():
    """A fully valid vision response used as the baseline in these tests."""
    return {
        "direction_bias": "UP",
        "trend": "up",
        "structure": "higher_highs_higher_lows",
        "support": [1.0, 1.05],
        "resistance": [1.2, 1.3],
        "breakout_probability": 0.6,
        "reversal_probability": 0.2,
        "momentum": "strong_up",
        "volatility": "normal",
        "confidence": 0.7,
        "invalidating_conditions": ["a close below 1.0 invalidates the bias"],
        "reasoning_summary": "clean bullish structure",
    }


def _schema_type_ok(schema, value):
    """Minimal JSON-schema type check for the subset used by the schema."""
    expected = schema.get("type")
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items", {})
        return all(_schema_type_ok(item_schema, v) for v in value)
    return True


class VisionSchemaTests(unittest.TestCase):
    def test_valid_payload_passes_validation(self):
        ok, errors = validate_vision_payload(_valid_payload())
        self.assertTrue(ok, msg=f"unexpected errors: {errors}")
        self.assertEqual(errors, [])

    def test_valid_payload_round_trips(self):
        payload = _valid_payload()
        result = coerce_payload(payload)
        self.assertIsInstance(result, VisionResult)
        self.assertEqual(result.direction_bias, Direction.UP)
        # A round-trip through to_dict/from_dict must be stable.
        again = VisionResult.from_dict(result.to_dict())
        self.assertEqual(again.to_dict(), result.to_dict())

    def test_to_dict_matches_schema(self):
        result = coerce_payload(_valid_payload())
        as_dict = result.to_dict()
        for key in VISION_RESPONSE_SCHEMA["required"]:
            self.assertIn(key, as_dict, msg=f"to_dict missing schema field {key}")
            prop = VISION_RESPONSE_SCHEMA["properties"][key]
            self.assertTrue(
                _schema_type_ok(prop, as_dict[key]),
                msg=f"field {key} has wrong type: {as_dict[key]!r}",
            )
        # Bookkeeping fields are also present.
        for key in ("provider", "degraded", "raw"):
            self.assertIn(key, as_dict)

    def test_missing_key_is_rejected_and_coerced(self):
        payload = _valid_payload()
        del payload["trend"]
        ok, errors = validate_vision_payload(payload)
        self.assertFalse(ok)
        self.assertTrue(any("trend" in e for e in errors))
        result = coerce_payload(payload)
        self.assertEqual(result.trend, "unknown")

    def test_wrong_type_confidence_is_rejected_and_raises(self):
        payload = _valid_payload()
        payload["confidence"] = "not-a-number"
        ok, errors = validate_vision_payload(payload)
        self.assertFalse(ok)
        self.assertTrue(any("confidence" in e for e in errors))
        with self.assertRaises(ValueError):
            coerce_payload(payload)

    def test_probability_above_one_is_rejected_and_clamped(self):
        payload = _valid_payload()
        payload["breakout_probability"] = 1.5
        ok, errors = validate_vision_payload(payload)
        self.assertFalse(ok)
        self.assertTrue(any("breakout_probability" in e for e in errors))
        result = coerce_payload(payload)
        self.assertEqual(result.breakout_probability, 1.0)

    def test_sideways_direction_is_rejected_and_coerced_to_neutral(self):
        payload = _valid_payload()
        payload["direction_bias"] = "SIDEWAYS"
        ok, errors = validate_vision_payload(payload)
        self.assertFalse(ok)
        self.assertTrue(any("direction_bias" in e for e in errors))
        result = coerce_payload(payload)
        self.assertEqual(result.direction_bias, Direction.NEUTRAL)

    def test_string_number_is_coerced(self):
        payload = _valid_payload()
        payload["confidence"] = "0.42"
        ok, _ = validate_vision_payload(payload)
        self.assertFalse(ok)  # strings are not valid numbers
        result = coerce_payload(payload)
        self.assertAlmostEqual(result.confidence, 0.42)

    def test_scalar_support_is_coerced_to_list(self):
        payload = _valid_payload()
        payload["support"] = 1.05
        result = coerce_payload(payload)
        self.assertEqual(result.support, [1.05])

    def test_non_mapping_payload_is_rejected(self):
        ok, errors = validate_vision_payload(["not", "a", "mapping"])
        self.assertFalse(ok)
        self.assertTrue(errors)
        with self.assertRaises(ValueError):
            coerce_payload(["not", "a", "mapping"])

    def test_text_field_given_list_raises(self):
        payload = _valid_payload()
        payload["trend"] = ["up", "down"]
        with self.assertRaises(ValueError):
            coerce_payload(payload)


if __name__ == "__main__":
    unittest.main()
