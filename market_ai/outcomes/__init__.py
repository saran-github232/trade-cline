"""Outcome resolution."""

from .resolver import (
    DEFAULT_AMBIGUITY_THRESHOLD,
    OutcomeResolver,
    ResolutionWindow,
    classify_outcome,
)

__all__ = ["OutcomeResolver", "ResolutionWindow", "classify_outcome", "DEFAULT_AMBIGUITY_THRESHOLD"]
