"""Structured logging and error-handling helpers.

Logs are emitted as single-line JSON so they can be grepped, shipped, or
parsed without a logging stack.  Credentials are scrubbed defensively: any
key whose name looks secret is replaced before it reaches a sink.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional

_SECRET_HINTS = (
    "key", "token", "secret", "password", "passwd", "credential",
    "authorization", "auth", "cookie", "session",
)


def scrub(payload: Any, _depth: int = 0) -> Any:
    """Recursively redact anything that looks like a secret.

    Values are replaced with ``"***"`` when their *key* contains a known
    secret hint.  This runs on every log record so a stray ``api_key`` cannot
    leak into a log file or the API surface.
    """
    if _depth > 6:
        return "<max-depth>"
    if isinstance(payload, Mapping):
        out: Dict[str, Any] = {}
        for key, value in payload.items():
            lkey = str(key).lower()
            if any(hint in lkey for hint in _SECRET_HINTS):
                out[str(key)] = "***"
            else:
                out[str(key)] = scrub(value, _depth + 1)
        return out
    if isinstance(payload, (list, tuple, set)):
        return [scrub(v, _depth + 1) for v in payload]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    return repr(payload)


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "context", None)
        if isinstance(extra, Mapping):
            payload["context"] = scrub(dict(extra))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(payload, default=str, separators=(",", ":"))


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger writing JSON to stderr (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        resolved = level or os.environ.get("MARKET_AI_LOG_LEVEL", "INFO")
        logger.setLevel(getattr(logging, str(resolved).upper(), logging.INFO))
    return logger


def log_event(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Emit a structured event with redacted context."""
    logger.log(level, message, extra={"context": scrub(context)})


@contextmanager
def timed(logger: logging.Logger, label: str, **context: Any) -> Iterator[Dict[str, Any]]:
    """Context manager logging duration and outcome of a block.

    Yields a mutable dict; add ``ok``/``error`` keys inside the block to
    enrich the emitted record.
    """
    started = time.perf_counter()
    result: Dict[str, Any] = {}
    try:
        yield result
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        log_event(
            logger, logging.ERROR, f"{label} failed",
            duration_ms=round(elapsed, 2), error=f"{type(exc).__name__}: {exc}",
            trace=traceback.format_exc(limit=3), **context,
        )
        raise
    else:
        elapsed = (time.perf_counter() - started) * 1000.0
        log_event(logger, logging.INFO, f"{label} ok", duration_ms=round(elapsed, 2), **context, **result)
