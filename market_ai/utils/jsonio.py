"""Safe JSON I/O helpers.

Standard ``json`` refuses to write NaN/Infinity by default and silently
accepts them on read, which would let non-finite floats corrupt the
prediction log.  These helpers normalise non-finite values to ``null`` on the
way out and convert them back to ``float('nan')`` on the way in.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

__all__ = ["dumps", "loads", "dump_json", "load_json", "atomic_write_text", "atomic_write_json", "sanitise"]

PathLike = Union[str, os.PathLike]


def sanitise(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` and coerce types."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitise(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if hasattr(obj, "to_dict"):
        return sanitise(obj.to_dict())
    return str(obj)


def dumps(obj: Any, *, indent: Optional[int] = None, sort_keys: bool = False) -> str:
    """JSON-encode ``obj`` with non-finite floats converted to ``null``."""
    return json.dumps(sanitise(obj), indent=indent, sort_keys=sort_keys, default=str, allow_nan=False)


def loads(text: str) -> Any:
    """JSON-decode, tolerating a UTF-8 BOM and blank input."""
    text = text.lstrip("\ufeff").strip()
    if not text:
        return None
    return json.loads(text)


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text atomically (temp file + rename) so readers never see a partial file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-", suffix=target.suffix)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def atomic_write_json(path: PathLike, obj: Any, *, indent: int = 2) -> Path:
    return atomic_write_text(path, dumps(obj, indent=indent, sort_keys=False) + "\n")


def dump_json(path: PathLike, obj: Any, *, indent: int = 2) -> Path:
    """Write JSON to ``path`` atomically."""
    return atomic_write_json(path, obj, indent=indent)


def load_json(path: PathLike, default: Any = None) -> Any:
    """Read JSON from ``path``; return ``default`` when missing or corrupt."""
    target = Path(path)
    if not target.exists():
        return default
    try:
        return loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
