"""Reasoning-supervisor subsystem.

Re-exports the public surface:

* :class:`ReasoningResult` - the supervisor's critique object;
* :class:`ReasoningSupervisor` - the orchestrator with a safe offline default.

Additional helpers (``validate_reasoning_payload``, ``OfflineReasoningSupervisor``
and ``ReasoningProvider``) are exported for tests and integrators.
"""

from __future__ import annotations

from .offline import OfflineReasoningSupervisor
from .schema import (
    ReasoningResult,
    coerce_reasoning_payload,
    validate_reasoning_payload,
)
from .supervisor import ReasoningProvider, ReasoningSupervisor

__all__ = [
    "ReasoningResult",
    "ReasoningSupervisor",
    "validate_reasoning_payload",
    "coerce_reasoning_payload",
    "OfflineReasoningSupervisor",
    "ReasoningProvider",
]

