"""Multi-signal ensemble and validation-based weighting."""

from .engine import (
    MODIFIER_PROVIDERS,
    WEIGHTED_PROVIDERS,
    EnsembleEngine,
    EnsembleResult,
    agreement_score,
    disagreement_score,
)
from .weights import ValidationWeightOptimizer, WeightCandidate, evaluate_weights

__all__ = [
    "EnsembleEngine", "EnsembleResult", "agreement_score", "disagreement_score",
    "WEIGHTED_PROVIDERS", "MODIFIER_PROVIDERS",
    "ValidationWeightOptimizer", "WeightCandidate", "evaluate_weights",
]
