"""Chart-vision subsystem.

Re-exports the public surface used by the rest of the system:

* :class:`VisionResult` - the normalised evidence object;
* :class:`ChartVisionAnalyser` - the orchestrator with offline fallback;
* :func:`validate_vision_payload` - the schema gate.

Additional helpers (``coerce_payload``, ``OfflineChartAnalyser``, the provider
adapters, ``VISION_RESPONSE_SCHEMA`` and ``render_chart_svg``) are exported for
tests and the UI.
"""

from __future__ import annotations

from .analyser import SUPPORTED_IMAGE_EXTENSIONS, ChartVisionAnalyser
from .chart_render import render_chart_svg
from .offline import OfflineChartAnalyser
from .providers import (
    AnthropicProvider,
    NullProvider,
    OpenAICompatibleProvider,
    VisionProvider,
    VisionProviderError,
    build_provider,
    register_provider,
)
from .schema import (
    DIRECTION_ALIASES,
    VISION_RESPONSE_SCHEMA,
    VisionResult,
    coerce_payload,
    validate_vision_payload,
)

__all__ = [
    "VisionResult",
    "ChartVisionAnalyser",
    "validate_vision_payload",
    "coerce_payload",
    "VISION_RESPONSE_SCHEMA",
    "DIRECTION_ALIASES",
    "OfflineChartAnalyser",
    "render_chart_svg",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "VisionProvider",
    "VisionProviderError",
    "NullProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "build_provider",
    "register_provider",
]

