"""Model registry: versioning, champion/challenger roles, rollback.

Design notes
------------
* Model artifacts are stored as **JSON**, never pickle.  Loading a model can
  therefore never execute arbitrary code, and an artifact is diffable in git.
* The registry keeps an append-only promotion history so that "what was
  champion at time T" is always answerable.
* Only one model may hold the ``champion`` role at a time.  Promotion is
  transactional: the old champion is archived in the same operation that
  installs the new one, so a crash cannot leave two champions or zero.
* ``rollback`` restores a previously archived version and is always available,
  because a promotion that cannot be undone is not a controlled promotion.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from ..utils.jsonio import atomic_write_json, load_json
from ..utils.logging import get_logger

__all__ = ["ModelRecord", "ModelRegistry", "ROLE_CHAMPION", "ROLE_CHALLENGER", "ROLE_ARCHIVED"]

ROLE_CHAMPION = "champion"
ROLE_CHALLENGER = "challenger"
ROLE_ARCHIVED = "archived"

PathLike = Union[str, Path]
_log = get_logger(__name__)


@dataclass
class ModelRecord:
    """Everything needed to reproduce and audit one trained model."""

    model_version: str
    role: str = ROLE_ARCHIVED
    model_name: str = "unknown"
    params: Dict[str, Any] = field(default_factory=dict)
    feature_version: str = "unknown"
    dataset_version: str = "unknown"
    training_period: Tuple[int, int] = (0, 0)
    validation_period: Tuple[int, int] = (0, 0)
    test_period: Tuple[int, int] = (0, 0)
    metrics: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    artifact_path: str = ""
    notes: str = ""
    parent_version: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = int(time.time())
        # Tuples survive JSON as lists; normalise on the way in.
        for attr in ("training_period", "validation_period", "test_period"):
            value = getattr(self, attr)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                setattr(self, attr, (int(value[0]), int(value[1])))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "role": self.role,
            "model_name": self.model_name,
            "params": dict(self.params),
            "feature_version": self.feature_version,
            "dataset_version": self.dataset_version,
            "training_period": list(self.training_period),
            "validation_period": list(self.validation_period),
            "test_period": list(self.test_period),
            "metrics": dict(self.metrics),
            "calibration": dict(self.calibration),
            "created_at": self.created_at,
            "artifact_path": self.artifact_path,
            "notes": self.notes,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelRecord":
        return cls(
            model_version=str(payload["model_version"]),
            role=str(payload.get("role", ROLE_ARCHIVED)),
            model_name=str(payload.get("model_name", "unknown")),
            params=dict(payload.get("params", {})),
            feature_version=str(payload.get("feature_version", "unknown")),
            dataset_version=str(payload.get("dataset_version", "unknown")),
            training_period=tuple(payload.get("training_period", (0, 0))),
            validation_period=tuple(payload.get("validation_period", (0, 0))),
            test_period=tuple(payload.get("test_period", (0, 0))),
            metrics=dict(payload.get("metrics", {})),
            calibration=dict(payload.get("calibration", {})),
            created_at=int(payload.get("created_at", 0)),
            artifact_path=str(payload.get("artifact_path", "")),
            notes=str(payload.get("notes", "")),
            parent_version=payload.get("parent_version"),
        )

    def summary(self) -> Dict[str, Any]:
        """Compact view for the dashboard."""
        return {
            "model_version": self.model_version,
            "role": self.role,
            "model_name": self.model_name,
            "feature_version": self.feature_version,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "oos_precision": (self.metrics or {}).get("test_precision"),
            "oos_expectancy": (self.metrics or {}).get("test_expectancy"),
            "calibration_error": (self.calibration or {}).get("ece"),
        }


class ModelRegistry:
    """Filesystem + SQLite backed registry of model versions.

    Layout::

        <root>/
            index.json              # all ModelRecords
            history.json            # append-only promotion log
            artifacts/
                <model_version>.json   # serialised model
    """

    INDEX_FILE = "index.json"
    HISTORY_FILE = "history.json"
    ARTIFACT_DIR = "artifacts"

    def __init__(self, root: PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self.ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / self.INDEX_FILE
        self._history_path = self.root / self.HISTORY_FILE
        self._index: Dict[str, ModelRecord] = {}
        self._history: List[Dict[str, Any]] = []
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        raw = load_json(self._index_path, default={}) or {}
        self._index = {}
        for version, payload in raw.items():
            try:
                self._index[version] = ModelRecord.from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                _log.warning("skipping corrupt registry entry %s: %s", version, exc)
        self._history = load_json(self._history_path, default=[]) or []

    def _flush(self) -> None:
        atomic_write_json(
            self._index_path, {v: r.to_dict() for v, r in self._index.items()}
        )
        atomic_write_json(self._history_path, self._history)

    def _log_history(self, action: str, *, from_version: Optional[str],
                     to_version: Optional[str], reason: str = "",
                     payload: Optional[Mapping[str, Any]] = None) -> None:
        self._history.append(
            {
                "at": int(time.time()),
                "action": action,
                "from_version": from_version,
                "to_version": to_version,
                "reason": reason,
                "payload": dict(payload or {}),
            }
        )

    # -- artifact handling -------------------------------------------------
    def _artifact_path(self, model_version: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_version)
        return self.root / self.ARTIFACT_DIR / f"{safe}.json"

    def _serialize_model(self, model: Any) -> Dict[str, Any]:
        """Serialise a model to plain JSON-compatible structures.

        We support an explicit ``to_dict()`` contract, which every model in
        ``market_ai.models`` implements.  Falling back to ``repr`` keeps the
        registry usable with third-party adapters without ever using pickle.
        """
        if hasattr(model, "to_dict"):
            payload = model.to_dict()
            return {"kind": "structured", "payload": payload}
        if isinstance(model, Mapping):
            return {"kind": "mapping", "payload": dict(model)}
        return {"kind": "opaque", "payload": {"repr": repr(model)}}

    def _deserialize_model(self, blob: Mapping[str, Any]) -> Any:
        kind = blob.get("kind")
        payload = blob.get("payload")
        if kind == "structured":
            from .adapters import rebuild_from_dict  # local import: avoid cycles
            return rebuild_from_dict(payload)
        if kind == "mapping":
            return dict(payload or {})
        return None

    # -- public API --------------------------------------------------------
    def save(self, record: ModelRecord, model: Any) -> str:
        """Persist a model artifact plus its metadata.  Returns the version."""
        path = self._artifact_path(record.model_version)
        atomic_write_json(
            path,
            {
                "model_version": record.model_version,
                "model_name": record.model_name,
                "feature_version": record.feature_version,
                "dataset_version": record.dataset_version,
                "params": record.params,
                "model": self._serialize_model(model),
            },
        )
        record.artifact_path = str(path)
        self._index[record.model_version] = record
        self._log_history(
            "save", from_version=None, to_version=record.model_version,
            reason=record.notes or "registered", payload=record.summary(),
        )
        self._flush()
        return record.model_version

    def load(self, model_version: str) -> Any:
        """Load a model artifact.  Returns ``None`` when unavailable."""
        blob = load_json(self._artifact_path(model_version))
        if not blob:
            return None
        return self._deserialize_model(blob.get("model", {}))

    def get(self, model_version: str) -> Optional[ModelRecord]:
        return self._index.get(model_version)

    def list(self, *, role: Optional[str] = None) -> List[ModelRecord]:
        records = list(self._index.values())
        if role:
            records = [r for r in records if r.role == role]
        return sorted(records, key=lambda r: r.created_at)

    def champion(self) -> Optional[ModelRecord]:
        """The single model currently in production, if any."""
        champs = [r for r in self._index.values() if r.role == ROLE_CHAMPION]
        if not champs:
            return None
        return max(champs, key=lambda r: r.created_at)

    def challengers(self) -> List[ModelRecord]:
        return self.list(role=ROLE_CHALLENGER)

    def register_challenger(self, record: ModelRecord, model: Any) -> str:
        """Register a candidate model.  It does NOT become production."""
        record.role = ROLE_CHALLENGER
        return self.save(record, model)

    def set_champion(self, model_version: str, *, reason: str = "") -> None:
        """Install ``model_version`` as the sole champion, archiving the old one.

        The whole operation is written atomically so a crash cannot leave the
        registry with zero or two champions.
        """
        target = self._index.get(model_version)
        if target is None:
            raise KeyError(f"unknown model_version {model_version!r}")
        previous = self.champion()
        if previous is not None and previous.model_version == model_version:
            return
        if previous is not None:
            previous.role = ROLE_ARCHIVED
        target.role = ROLE_CHAMPION
        self._log_history(
            "promote",
            from_version=previous.model_version if previous else None,
            to_version=model_version,
            reason=reason or "passed promotion gates",
            payload=target.summary(),
        )
        self._flush()
        _log.info("promoted %s to champion (previous=%s)", model_version,
                  previous.model_version if previous else None)

    def rollback(self, to_version: Optional[str] = None) -> Optional[ModelRecord]:
        """Restore a previous champion.

        With no argument, rolls back to the most recently archived model that
        actually held the champion role (derived from the promotion history).
        """
        target: Optional[ModelRecord] = None
        if to_version:
            target = self._index.get(to_version)
            if target is None:
                raise KeyError(f"unknown model_version {to_version!r}")
        else:
            for entry in reversed(self._history):
                if entry.get("action") == "promote" and entry.get("from_version"):
                    candidate = self._index.get(entry["from_version"])
                    if candidate is not None:
                        target = candidate
                        break
        if target is None:
            return None
        current = self.champion()
        if current is not None:
            current.role = ROLE_ARCHIVED
        target.role = ROLE_CHAMPION
        self._log_history(
            "rollback",
            from_version=current.model_version if current else None,
            to_version=target.model_version,
            reason="manual rollback",
            payload=target.summary(),
        )
        self._flush()
        return target

    def history(self) -> List[Dict[str, Any]]:
        """Append-only audit log, oldest first."""
        return list(self._history)

    def status(self) -> Dict[str, Any]:
        """Dashboard view of champion/challenger state."""
        champ = self.champion()
        chals = self.challengers()
        last_promo = next(
            (e for e in reversed(self._history) if e.get("action") in ("promote", "rollback")),
            None,
        )
        return {
            "champion": champ.summary() if champ else None,
            "challenger": chals[-1].summary() if chals else None,
            "n_models": len(self._index),
            "last_promotion": last_promo,
            "rollback_version": (last_promo or {}).get("from_version"),
            "history": self._history[-20:],
        }
