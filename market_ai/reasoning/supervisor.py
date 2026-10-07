"""Reasoning-supervisor orchestration.

:class:`ReasoningSupervisor` is the single entry point for the reasoning stage.
It defaults to the deterministic :class:`OfflineReasoningSupervisor` and may be
given an optional external :class:`ReasoningProvider` (an LLM critique).

Whatever the source, the supervisor enforces the safety contract *after* the
critique is produced:

* ``recommendation`` may only be the base decision or ``NO_TRADE`` - an
  external provider can never create or upgrade a directional call;
* ``confidence_adjustment`` is clamped into ``[-1, 0]``;
* any provider failure or malformed response falls back to the offline
  supervisor with ``degraded=True``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from ..config import Config
from ..types import Decision
from ..utils.logging import get_logger, log_event
from .offline import OfflineReasoningSupervisor, _coerce_decision
from .schema import (
    ReasoningResult,
    coerce_reasoning_payload,
    validate_reasoning_payload,
)

__all__ = ["ReasoningSupervisor", "ReasoningProvider"]

_LOGGER = get_logger("market_ai.reasoning.supervisor")


@runtime_checkable
class ReasoningProvider(Protocol):
    """A source of structured reasoning critiques."""

    name: str

    def review(self, *, context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a raw (pre-validation) reasoning payload mapping."""
        ...


class ReasoningSupervisor:
    """Orchestrate the reasoning stage with a safe offline default."""

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        provider: Optional[ReasoningProvider] = None,
    ) -> None:
        """Store configuration and an optional external provider.

        The keyword-only ``provider`` is an *extension* of the frozen
        ``ReasoningSupervisor(config)`` signature: it is optional and does not
        change how positional callers construct the object.  It exists so an
        external provider can be injected (and, in tests, a deliberately
        broken one).
        """
        self.config = config or Config()
        self._offline = OfflineReasoningSupervisor(self.config)
        self._provider = provider
        #: The normalised result of the last call.
        self.last_result: Optional[ReasoningResult] = None

    def review(self, *, context: Mapping[str, Any]) -> ReasoningResult:
        """Review ``context`` and return a safety-constrained result."""
        base = _coerce_decision(
            (context or {}).get("decision", (context or {}).get("base_decision"))
        )

        if self._provider is None:
            result = self._offline.review(context=context)
            self.last_result = result
            return result

        provider_name = getattr(self._provider, "name", "external")
        try:
            raw = self._provider.review(context=context)
        except Exception as exc:  # noqa: BLE001 - a provider must never crash us
            return self._degrade(
                context, base, f"provider error: {type(exc).__name__}: {exc}"
            )

        if not isinstance(raw, Mapping):
            return self._degrade(context, base, "provider returned a non-mapping")

        ok, errors = validate_reasoning_payload(raw)
        if not ok:
            return self._degrade(
                context, base, "malformed provider response: " + "; ".join(errors)
            )

        try:
            coerced = coerce_reasoning_payload(raw)
        except ValueError as exc:
            return self._degrade(context, base, f"uncoercible provider response: {exc}")

        result = replace(coerced, provider=str(provider_name), degraded=False)
        result = self._enforce_safety(result, base)
        self.last_result = result
        return result

    # -- internals --------------------------------------------------------
    def _degrade(
        self,
        context: Mapping[str, Any],
        base: Optional[Decision],
        reason: str,
    ) -> ReasoningResult:
        """Fall back to the offline supervisor and mark the result degraded."""
        offline = self._offline.review(context=context)
        degraded = replace(
            offline,
            degraded=True,
            explanation=f"{offline.explanation} (fell back to offline: {reason})",
        )
        degraded = self._enforce_safety(degraded, base)
        log_event(_LOGGER, 30, "reasoning fell back to offline supervisor", reason=reason)
        self.last_result = degraded
        return degraded

    @staticmethod
    def _enforce_safety(
        result: ReasoningResult, base: Optional[Decision]
    ) -> ReasoningResult:
        """Guarantee the supervisor can only reduce, never upgrade.

        The recommendation is clamped to ``{base, NO_TRADE}`` and the
        confidence adjustment to ``[-1, 0]`` regardless of what a provider
        returned.
        """
        recommendation = result.recommendation
        if base is None or base == Decision.NO_TRADE:
            recommendation = Decision.NO_TRADE
        elif recommendation not in (base, Decision.NO_TRADE):
            # The provider tried to invent/upgrade a directional call: ignore it.
            recommendation = base
        adjustment = -abs(float(result.confidence_adjustment))
        adjustment = max(-1.0, min(0.0, adjustment))
        return replace(
            result, recommendation=recommendation, confidence_adjustment=adjustment
        )
