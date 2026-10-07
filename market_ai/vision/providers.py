"""Optional multimodal vision providers.

These adapters talk to external multimodal APIs (OpenAI-compatible and
Anthropic).  They are strictly optional:

* credentials are read **only** from environment variables - never hard-coded,
  never persisted, never logged (``utils.logging.scrub`` redacts them);
* nothing performs I/O at import time; the standard-library ``urllib`` is used
  lazily inside :meth:`analyse`;
* every failure raises :class:`VisionProviderError`, which the orchestrator
  turns into a graceful fallback to the deterministic offline analyser.

The default provider is ``offline`` (see :mod:`market_ai.vision.offline`), so
the system works with no network and no API key.  None of this code evades,
spoofs or defeats any platform control - it simply calls a documented HTTP API
with a caller-supplied image.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..config import Config
from ..utils.jsonio import sanitise
from ..utils.logging import get_logger, log_event

__all__ = [
    "VisionProviderError",
    "VisionProvider",
    "NullProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "register_provider",
    "build_provider",
    "VISION_API_KEY_ENV",
]

#: Preferred env var; the vendor-specific names are accepted as fallbacks.
VISION_API_KEY_ENV = "MARKET_AI_VISION_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

_LOGGER = get_logger("market_ai.vision.providers")

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


class VisionProviderError(RuntimeError):
    """Raised when a vision provider cannot produce a valid response."""


@runtime_checkable
class VisionProvider(Protocol):
    """A source of structured chart-vision evidence."""

    name: str

    def analyse(
        self,
        *,
        image_path: Optional[str],
        summary: Mapping[str, Any],
        timeframe: Optional[Any] = None,
        asset: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Return a raw (pre-validation) vision payload mapping."""
        ...


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _read_api_key(env_names: Sequence[str]) -> Optional[str]:
    """Return the first non-empty credential found in the environment."""
    for name in env_names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _build_prompt(summary: Mapping[str, Any], timeframe: Any, asset: Any) -> str:
    """Build the instruction + structured-summary prompt sent to the model."""
    context = {
        "asset": asset,
        "timeframe": getattr(timeframe, "value", timeframe),
        "summary": sanitise(dict(summary or {})),
    }
    return (
        "You are a chart-reading assistant. Analyse the supplied candlestick "
        "image and respond with STRICT JSON only, matching exactly these keys: "
        "direction_bias (UP|DOWN|NEUTRAL), trend, structure, support (array of "
        "numbers), resistance (array of numbers), breakout_probability (0..1), "
        "reversal_probability (0..1), momentum, volatility, confidence (0..1), "
        "invalidating_conditions (array of strings), reasoning_summary. "
        "Use the numerical summary below as supporting evidence; never invent "
        "price levels that are not visible. Summary: "
        + json.dumps(context, default=str, separators=(",", ":"))
    )


def _encode_image(image_path: str) -> str:
    """Read an image file and return its base64 encoding."""
    try:
        with open(image_path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise VisionProviderError(f"cannot read image {image_path!r}: {exc}") from exc
    if not data:
        raise VisionProviderError(f"image {image_path!r} is empty")
    return base64.b64encode(data).decode("ascii")


def _media_type(image_path: str) -> str:
    """Best-effort media type from the file extension (defaults to PNG)."""
    lowered = image_path.lower()
    for ext, mime in _MEDIA_TYPES.items():
        if lowered.endswith(ext):
            return mime
    return "image/png"


def _post_json(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> Mapping[str, Any]:
    """POST ``payload`` as JSON and decode the JSON response.

    Uses ``urllib.request`` from the standard library.  Any transport or
    decoding error is normalised into :class:`VisionProviderError` so callers
    only ever deal with one failure type.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - needs network
        raise VisionProviderError(f"HTTP error {exc.code} from {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:  # pragma: no cover
        raise VisionProviderError(f"transport error talking to {url}: {exc}") from exc
    try:
        decoded = json.loads(raw)
    except ValueError as exc:  # pragma: no cover - needs network
        raise VisionProviderError("provider returned non-JSON body") from exc
    if not isinstance(decoded, Mapping):
        raise VisionProviderError("provider returned a non-object JSON body")
    return decoded


def _with_retries(
    operation: Callable[[], Mapping[str, Any]],
    *,
    max_retries: int,
    base_delay: float = 0.5,
) -> Mapping[str, Any]:
    """Run ``operation`` with exponential backoff, up to ``max_retries`` times."""
    attempt = 0
    while True:
        try:
            return operation()
        except VisionProviderError as exc:
            if attempt >= max(0, max_retries):
                raise
            delay = base_delay * (2 ** attempt)
            log_event(
                _LOGGER,
                30,  # logging.WARNING without importing logging just for a level
                "vision provider attempt failed; retrying",
                attempt=attempt + 1,
                delay=round(delay, 3),
                error=str(exc),
            )
            time.sleep(delay)
            attempt += 1


def _extract_json_object(text: str) -> Mapping[str, Any]:
    """Parse a JSON object from model text, tolerating code-fence wrapping."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        # Drop a leading ```json fence and the trailing fence if present.
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    cleaned = cleaned.strip()
    try:
        decoded = json.loads(cleaned)
    except ValueError as exc:
        raise VisionProviderError("model response was not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise VisionProviderError("model response was not a JSON object")
    return decoded



# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
class NullProvider:
    """Provider used when no external credentials exist.

    It always raises :class:`VisionProviderError`; the orchestrator catches
    that and falls back to the offline analyser with ``degraded=True``.
    """

    name = "null"

    def __init__(self, reason: str = "no vision provider is configured") -> None:
        """Store the human-readable reason reported on failure."""
        self._reason = reason

    def analyse(
        self,
        *,
        image_path: Optional[str] = None,
        summary: Optional[Mapping[str, Any]] = None,
        timeframe: Optional[Any] = None,
        asset: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Always raise: there is no configured provider."""
        raise VisionProviderError(self._reason)


class OpenAICompatibleProvider:
    """Adapter for OpenAI-compatible ``/chat/completions`` endpoints."""

    name = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        api_key_envs: Sequence[str] = (VISION_API_KEY_ENV, OPENAI_API_KEY_ENV),
        timeout: float = 20.0,
        max_retries: int = 2,
    ) -> None:
        """Configure the endpoint.  No I/O or credential lookup happens here."""
        self._model = model
        self._base_url = base_url
        self._api_key_envs = tuple(api_key_envs)
        self._timeout = float(timeout)
        self._max_retries = int(max_retries)

    @property
    def available(self) -> bool:
        """True when a credential is present in the environment."""
        return _read_api_key(self._api_key_envs) is not None

    def analyse(
        self,
        *,
        image_path: Optional[str],
        summary: Mapping[str, Any],
        timeframe: Optional[Any] = None,
        asset: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Send the image + prompt and return the parsed JSON object."""
        key = _read_api_key(self._api_key_envs)
        if key is None:
            raise VisionProviderError(
                "openai provider has no credentials in the environment"
            )
        if not image_path:
            raise VisionProviderError("openai provider requires an image_path")

        b64 = _encode_image(image_path)
        prompt = _build_prompt(summary, timeframe, asset)
        payload = {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{_media_type(image_path)};base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        def _call() -> Mapping[str, Any]:
            data = _post_json(self._base_url, headers, payload, self._timeout)
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise VisionProviderError("openai response contained no choices")
            message = choices[0].get("message", {}) if isinstance(choices[0], Mapping) else {}
            content = message.get("content")
            if not isinstance(content, str):
                raise VisionProviderError("openai response contained no text content")
            return _extract_json_object(content)

        return _with_retries(_call, max_retries=self._max_retries)



class AnthropicProvider:
    """Adapter for the Anthropic ``/v1/messages`` endpoint."""

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-3-5-sonnet-latest",
        base_url: str = "https://api.anthropic.com/v1/messages",
        api_key_envs: Sequence[str] = (VISION_API_KEY_ENV, ANTHROPIC_API_KEY_ENV),
        timeout: float = 20.0,
        max_retries: int = 2,
        max_tokens: int = 1024,
    ) -> None:
        """Configure the endpoint.  No I/O or credential lookup happens here."""
        self._model = model
        self._base_url = base_url
        self._api_key_envs = tuple(api_key_envs)
        self._timeout = float(timeout)
        self._max_retries = int(max_retries)
        self._max_tokens = int(max_tokens)

    @property
    def available(self) -> bool:
        """True when a credential is present in the environment."""
        return _read_api_key(self._api_key_envs) is not None

    def analyse(
        self,
        *,
        image_path: Optional[str],
        summary: Mapping[str, Any],
        timeframe: Optional[Any] = None,
        asset: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Send the image + prompt and return the parsed JSON object."""
        key = _read_api_key(self._api_key_envs)
        if key is None:
            raise VisionProviderError(
                "anthropic provider has no credentials in the environment"
            )
        if not image_path:
            raise VisionProviderError("anthropic provider requires an image_path")

        b64 = _encode_image(image_path)
        prompt = _build_prompt(summary, timeframe, asset)
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _media_type(image_path),
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        def _call() -> Mapping[str, Any]:
            data = _post_json(self._base_url, headers, payload, self._timeout)
            blocks = data.get("content")
            if not isinstance(blocks, list) or not blocks:
                raise VisionProviderError("anthropic response contained no content")
            text = None
            for block in blocks:
                if isinstance(block, Mapping) and block.get("type") == "text":
                    text = block.get("text")
                    break
            if not isinstance(text, str):
                raise VisionProviderError("anthropic response contained no text block")
            return _extract_json_object(text)

        return _with_retries(_call, max_retries=self._max_retries)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
#: Name -> factory(config) -> provider.  Tests and integrators can register
#: additional providers (e.g. a fake) without touching this module.
_REGISTRY: Dict[str, Callable[[Optional[Config]], VisionProvider]] = {}


def register_provider(
    name: str, factory: Callable[[Optional[Config]], VisionProvider]
) -> None:
    """Register a provider factory under ``name`` (case-insensitive)."""
    _REGISTRY[str(name).strip().lower()] = factory


def build_provider(name: str, config: Optional[Config] = None) -> VisionProvider:
    """Resolve a provider by name, defaulting to :class:`NullProvider`.

    Unknown names yield a :class:`NullProvider` so an external request without
    a usable backend degrades gracefully instead of crashing.
    """
    key = str(name or "").strip().lower()
    if key in _REGISTRY:
        return _REGISTRY[key](config)
    timeout = config.vision.timeout_seconds if config is not None else 20.0
    retries = config.vision.max_retries if config is not None else 2
    if key in ("openai", "openai_compatible", "azure_openai", "gpt"):
        return OpenAICompatibleProvider(timeout=timeout, max_retries=retries)
    if key in ("anthropic", "claude"):
        return AnthropicProvider(timeout=timeout, max_retries=retries)
    return NullProvider(f"unknown vision provider {name!r}")

