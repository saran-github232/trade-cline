"""Chart-vision orchestrator with provider selection and graceful fallback.

:class:`ChartVisionAnalyser` is the single entry point the rest of the system
talks to.  It:

* selects the configured provider (``offline`` by default);
* validates the image path before any external call;
* validates and normalises the model response against the frozen schema;
* falls back to the deterministic offline analyser on *any* failure, setting
  ``degraded=True`` so downstream consumers know the evidence is second-best;
* stores both the raw response and the normalised :class:`VisionResult`.

The vision layer is evidence, never authority: even a perfectly valid external
response is capped in influence elsewhere via ``config.vision.max_influence``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional

from ..config import Config
from ..types import MarketSeries, Timeframe
from ..utils.logging import get_logger, log_event
from .offline import OfflineChartAnalyser
from .providers import VisionProviderError, build_provider
from .schema import VisionResult, coerce_payload, validate_vision_payload

__all__ = ["ChartVisionAnalyser", "SUPPORTED_IMAGE_EXTENSIONS"]

#: Extensions we accept for a caller-supplied chart image.
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}
)

_LOGGER = get_logger("market_ai.vision.analyser")


class ChartVisionAnalyser:
    """Coordinate chart-vision analysis with a safe offline fallback."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Store configuration and build the offline analyser."""
        self.config = config or Config()
        self._offline = OfflineChartAnalyser(self.config)
        #: The raw provider response (or offline-derived payload) from the last
        #: call, retained for auditing and the UI.
        self.last_raw_response: Mapping[str, Any] = {}
        #: The normalised result of the last call.
        self.last_result: Optional[VisionResult] = None
        #: Human-readable reason the last call fell back (empty when it did not).
        self.last_fallback_reason: str = ""

    def analyse(
        self,
        *,
        series: MarketSeries,
        summary: Mapping[str, Any],
        image_path: Optional[str] = None,
        timeframe: Optional[Timeframe] = None,
        asset: Optional[str] = None,
    ) -> VisionResult:
        """Analyse a chart, returning a normalised :class:`VisionResult`.

        Falls back to the offline analyser (with ``degraded=True``) whenever an
        external provider is unavailable, unconfigured, given a bad image, or
        returns something that fails schema validation.
        """
        provider_name = (self.config.vision.provider or "offline").strip().lower()

        # The offline path is the default and never needs an image or network.
        if provider_name in ("", "offline", "none", "null"):
            result = self._offline.analyse(
                series=series, summary=summary, timeframe=timeframe, asset=asset
            )
            self._record(result)
            return result

        image_ok, image_reason = self._check_image(image_path)
        if not image_ok:
            return self._fallback(
                series, summary, timeframe, asset, reason=image_reason
            )

        provider = build_provider(provider_name, self.config)
        try:
            raw = provider.analyse(
                image_path=image_path, summary=summary,
                timeframe=timeframe, asset=asset,
            )
        except VisionProviderError as exc:
            return self._fallback(
                series, summary, timeframe, asset, reason=f"provider error: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 - never let a provider crash us
            return self._fallback(
                series, summary, timeframe, asset,
                reason=f"unexpected provider failure: {type(exc).__name__}: {exc}",
            )

        if not isinstance(raw, Mapping):
            return self._fallback(
                series, summary, timeframe, asset,
                reason="provider returned a non-mapping response",
            )

        ok, errors = validate_vision_payload(raw)
        if not ok:
            return self._fallback(
                series, summary, timeframe, asset,
                reason="malformed provider response: " + "; ".join(errors),
            )

        try:
            result = coerce_payload(raw)
        except ValueError as exc:
            return self._fallback(
                series, summary, timeframe, asset,
                reason=f"uncoercible provider response: {exc}",
            )

        result = replace(result, provider=provider_name, degraded=False)
        self._record(result)
        return result

    # -- internals --------------------------------------------------------
    def _check_image(self, image_path: Optional[str]) -> tuple[bool, str]:
        """Validate that ``image_path`` exists and has a supported extension."""
        if not image_path:
            return False, "external vision provider requires an image_path"
        path = Path(image_path)
        try:
            if not path.exists() or not path.is_file():
                return False, f"image not found: {image_path}"
        except OSError as exc:
            return False, f"image not readable: {exc}"
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return False, f"unsupported image extension: {path.suffix or '<none>'}"
        return True, ""

    def _fallback(
        self,
        series: MarketSeries,
        summary: Mapping[str, Any],
        timeframe: Optional[Timeframe],
        asset: Optional[str],
        *,
        reason: str,
    ) -> VisionResult:
        """Run the offline analyser and mark the result as degraded."""
        offline = self._offline.analyse(
            series=series, summary=summary, timeframe=timeframe, asset=asset
        )
        degraded = replace(offline, degraded=True)
        self.last_fallback_reason = reason
        log_event(
            _LOGGER,
            30,  # WARNING
            "vision fell back to offline analyser",
            reason=reason,
            asset=asset,
        )
        self._record(degraded)
        return degraded

    def _record(self, result: VisionResult) -> None:
        """Store the raw response and the normalised result for auditing."""
        self.last_result = result
        self.last_raw_response = dict(result.raw)

