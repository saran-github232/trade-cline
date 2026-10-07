"""SQLite-backed persistence (standard library only).

Design notes
------------
* Every table stores the canonical object as a JSON ``payload`` column *plus*
  the handful of columns we actually filter on.  This keeps the schema stable
  while the domain objects evolve, and it means a schema change never requires
  a data migration for the JSON body.
* Connections are per-thread because SQLite connection objects are not
  thread-safe.  The API server and the background retrainer can therefore run
  concurrently without corruption.
* WAL journalling is used so a long backtest writing thousands of rows never
  blocks the read path used by the dashboard.
* No ``pickle`` anywhere: loading a row can never execute arbitrary code.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from ..types import Experience, Outcome, Prediction
from ..utils.jsonio import dumps, loads

__all__ = ["Database", "Repository", "PredictionRepository", "OutcomeRepository",
           "ExperienceRepository", "ModelRepository", "BacktestRepository",
           "DatasetRepository", "EventRepository", "PaperTradeRepository"]

PathLike = Union[str, Path]

_SCHEMA: Sequence[str] = (
    """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id TEXT PRIMARY KEY,
        timestamp INTEGER NOT NULL,
        asset TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        decision TEXT NOT NULL,
        confidence REAL NOT NULL,
        regime TEXT NOT NULL,
        model_version TEXT NOT NULL,
        strategy_version TEXT NOT NULL,
        resolved INTEGER NOT NULL DEFAULT 0,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_pred_asset ON predictions(asset, timeframe)",
    "CREATE INDEX IF NOT EXISTS idx_pred_unresolved ON predictions(resolved, timestamp)",
    """
    CREATE TABLE IF NOT EXISTS outcomes (
        prediction_id TEXT PRIMARY KEY,
        resolved_at INTEGER NOT NULL,
        actual_direction TEXT NOT NULL,
        error_category TEXT NOT NULL,
        profit_or_loss REAL NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_out_resolved ON outcomes(resolved_at)",
    """
    CREATE TABLE IF NOT EXISTS experiences (
        prediction_id TEXT PRIMARY KEY,
        timestamp INTEGER NOT NULL,
        asset TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        regime TEXT NOT NULL,
        decision TEXT NOT NULL,
        error_category TEXT NOT NULL,
        profit_or_loss REAL NOT NULL,
        model_version TEXT NOT NULL,
        feature_version TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_exp_ts ON experiences(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_exp_regime ON experiences(regime)",
    "CREATE INDEX IF NOT EXISTS idx_exp_err ON experiences(error_category)",
    """
    CREATE TABLE IF NOT EXISTS models (
        model_version TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        model_name TEXT NOT NULL,
        feature_version TEXT NOT NULL,
        dataset_version TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_model_role ON models(role, created_at)",
    """
    CREATE TABLE IF NOT EXISTS model_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        at INTEGER NOT NULL,
        action TEXT NOT NULL,
        from_version TEXT,
        to_version TEXT,
        reason TEXT,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtests (
        backtest_id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        asset TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        model_version TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_version TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        feature_version TEXT NOT NULL,
        n_samples INTEGER NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        at INTEGER NOT NULL,
        kind TEXT NOT NULL,
        message TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_at ON events(at)",
    """
    CREATE TABLE IF NOT EXISTS paper_trades (
        trade_id TEXT PRIMARY KEY,
        prediction_id TEXT NOT NULL,
        opened_at INTEGER NOT NULL,
        asset TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        decision TEXT NOT NULL,
        stake REAL NOT NULL,
        pnl REAL NOT NULL DEFAULT 0.0,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_paper_opened ON paper_trades(opened_at)",
)


class Repository:
    """Generic JSON-payload repository over one table."""

    table: str = ""
    pk: str = "id"

    def __init__(self, db: "Database") -> None:
        self._db = db

    # -- internals ---------------------------------------------------------
    def _columns(self, obj: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    # -- public API --------------------------------------------------------
    def add(self, obj: Any) -> None:
        """Insert or replace one object."""
        payload = obj.to_dict() if hasattr(obj, "to_dict") else dict(obj)
        columns = self._columns(payload)
        columns["payload"] = dumps(payload)
        names = ", ".join(columns)
        marks = ", ".join("?" for _ in columns)
        sql = f"INSERT OR REPLACE INTO {self.table} ({names}) VALUES ({marks})"
        with self._db.cursor() as cur:
            cur.execute(sql, list(columns.values()))

    def add_many(self, items: Iterable[Any]) -> int:
        """Insert many objects in a single transaction.  Returns the count."""
        count = 0
        with self._db.cursor() as cur:
            for obj in items:
                payload = obj.to_dict() if hasattr(obj, "to_dict") else dict(obj)
                columns = self._columns(payload)
                columns["payload"] = dumps(payload)
                names = ", ".join(columns)
                marks = ", ".join("?" for _ in columns)
                cur.execute(
                    f"INSERT OR REPLACE INTO {self.table} ({names}) VALUES ({marks})",
                    list(columns.values()),
                )
                count += 1
        return count

    def get(self, key: Any) -> Optional[Dict[str, Any]]:
        """Return the raw payload dict for ``key`` or ``None``."""
        with self._db.cursor() as cur:
            cur.execute(f"SELECT payload FROM {self.table} WHERE {self.pk} = ?", (key,))
            row = cur.fetchone()
        return loads(row[0]) if row else None

    def list(self, *, limit: Optional[int] = None, order_by: str = None, **filters: Any) -> List[Dict[str, Any]]:
        """List payloads, optionally filtered on indexed columns."""
        where: List[str] = []
        params: List[Any] = []
        for key, value in filters.items():
            if value is None:
                continue
            where.append(f"{key} = ?")
            params.append(value)
        sql = f"SELECT payload FROM {self.table}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order_by or self.pk}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._db.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            value = loads(row[0])
            if value is not None:
                out.append(value)
        return out

    def count(self, **filters: Any) -> int:
        where: List[str] = []
        params: List[Any] = []
        for key, value in filters.items():
            if value is None:
                continue
            where.append(f"{key} = ?")
            params.append(value)
        sql = f"SELECT COUNT(*) FROM {self.table}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._db.cursor() as cur:
            cur.execute(sql, params)
            return int(cur.fetchone()[0])

    def delete(self, key: Any) -> bool:
        with self._db.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE {self.pk} = ?", (key,))
            return cur.rowcount > 0

    def clear(self) -> None:
        with self._db.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table}")


class PredictionRepository(Repository):
    """Stores every :class:`Prediction` ever made."""

    table = "predictions"
    pk = "prediction_id"

    def _columns(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "prediction_id": payload["prediction_id"],
            "timestamp": int(payload["timestamp"]),
            "asset": payload["asset"],
            "timeframe": payload["timeframe"],
            "decision": payload["decision"],
            "confidence": float(payload.get("confidence", 0.0)),
            "regime": payload.get("regime", "UNCERTAIN"),
            "model_version": payload.get("model_version", "unknown"),
            "strategy_version": payload.get("strategy_version", "unknown"),
            "resolved": 0,
        }

    def add_prediction(self, prediction: Prediction) -> None:
        self.add(prediction)

    def mark_resolved(self, prediction_id: str) -> None:
        with self._db.cursor() as cur:
            cur.execute("UPDATE predictions SET resolved = 1 WHERE prediction_id = ?", (prediction_id,))

    def unresolved(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Predictions whose outcome has not been recorded yet."""
        sql = "SELECT payload FROM predictions WHERE resolved = 0 ORDER BY timestamp"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._db.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [v for v in (loads(r[0]) for r in rows) if v is not None]

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        sql = "SELECT payload FROM predictions ORDER BY timestamp DESC LIMIT ?"
        with self._db.cursor() as cur:
            cur.execute(sql, (int(limit),))
            rows = cur.fetchall()
        return [v for v in (loads(r[0]) for r in rows) if v is not None]


class OutcomeRepository(Repository):
    """Stores realised outcomes keyed by prediction id."""

    table = "outcomes"
    pk = "prediction_id"

    def _columns(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "prediction_id": payload["prediction_id"],
            "resolved_at": int(payload["resolved_at"]),
            "actual_direction": payload["actual_direction"],
            "error_category": payload["error_category"],
            "profit_or_loss": float(payload.get("profit_or_loss", 0.0)),
        }

    def add_outcome(self, outcome: Outcome) -> None:
        self.add(outcome)


class ExperienceRepository(Repository):
    """The market-experience memory: completed prediction/outcome pairs."""

    table = "experiences"
    pk = "prediction_id"

    def _columns(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "prediction_id": payload["prediction_id"],
            "timestamp": int(payload["timestamp"]),
            "asset": payload["asset"],
            "timeframe": payload["timeframe"],
            "regime": payload.get("regime", "UNCERTAIN"),
            "decision": payload["decision"],
            "error_category": payload["error_category"],
            "profit_or_loss": float(payload.get("profit_or_loss", 0.0)),
            "model_version": payload.get("model_version", "unknown"),
            "feature_version": payload.get("feature_version", "unknown"),
        }

    def add_experience(self, experience: Experience) -> None:
        self.add(experience)

    def since(self, ts: int, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = "SELECT payload FROM experiences WHERE timestamp >= ? ORDER BY timestamp"
        params: List[Any] = [int(ts)]
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._db.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [v for v in (loads(r[0]) for r in rows) if v is not None]


class ModelRepository(Repository):
    """Registry of trained models and their roles."""

    table = "models"
    pk = "model_version"

    def _columns(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "model_version": payload["model_version"],
            "role": payload.get("role", "archived"),
            "model_name": payload.get("model_name", "unknown"),
            "feature_version": payload.get("feature_version", "unknown"),
            "dataset_version": payload.get("dataset_version", "unknown"),
            "created_at": int(payload.get("created_at", 0)),
        }

    def log_history(self, action: str, *, from_version: Optional[str], to_version: Optional[str],
                    reason: str = "", payload: Optional[Mapping[str, Any]] = None) -> None:
        """Append an immutable audit record to the promotion history."""
        with self._db.cursor() as cur:
            cur.execute(
                "INSERT INTO model_history (at, action, from_version, to_version, reason, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (int(time.time()), action, from_version, to_version, reason, dumps(dict(payload or {}))),
            )

    def history(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._db.cursor() as cur:
            cur.execute(
                "SELECT at, action, from_version, to_version, reason, payload "
                "FROM model_history ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()
        return [
            {
                "at": r[0], "action": r[1], "from_version": r[2],
                "to_version": r[3], "reason": r[4], "payload": loads(r[5]) or {},
            }
            for r in rows
        ]


class BacktestRepository(Repository):
    table = "backtests"
    pk = "backtest_id"

    def _columns(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "backtest_id": payload["backtest_id"],
            "created_at": int(payload.get("created_at", 0)),
            "asset": payload.get("asset", ""),
            "timeframe": payload.get("timeframe", ""),
            "model_version": payload.get("model_version", "unknown"),
        }


class DatasetRepository(Repository):
    table = "datasets"
    pk = "dataset_version"

    def _columns(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "dataset_version": payload["dataset_version"],
            "created_at": int(payload.get("created_at", 0)),
            "feature_version": payload.get("feature_version", "unknown"),
            "n_samples": int(payload.get("n_samples", 0)),
        }


class EventRepository(Repository):
    """Append-only operational event log (retraining triggers, failures, ...)."""

    table = "events"
    pk = "id"

    def _columns(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "at": int(payload.get("at", time.time())),
            "kind": payload.get("kind", "info"),
            "message": payload.get("message", ""),
        }

    def log(self, kind: str, message: str, **payload: Any) -> None:
        self.add({"at": int(time.time()), "kind": kind, "message": message, "payload": dict(payload)})

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._db.cursor() as cur:
            cur.execute("SELECT payload FROM events ORDER BY id DESC LIMIT ?", (int(limit),))
            rows = cur.fetchall()
        return [v for v in (loads(r[0]) for r in rows) if v is not None]


class PaperTradeRepository(Repository):
    table = "paper_trades"
    pk = "trade_id"

    def _columns(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "trade_id": payload["trade_id"],
            "prediction_id": payload.get("prediction_id", ""),
            "opened_at": int(payload.get("opened_at", 0)),
            "asset": payload.get("asset", ""),
            "timeframe": payload.get("timeframe", ""),
            "decision": payload.get("decision", "NO-TRADE"),
            "stake": float(payload.get("stake", 0.0)),
            "pnl": float(payload.get("pnl", 0.0)),
        }


class Database:
    """Thread-safe SQLite database with one repository per table.

    Usage::

        db = Database("artifacts/market_ai.db")
        db.initialize()
        db.predictions.add(prediction)
        db.experiences.count()
        db.close()

    A connection is opened lazily per thread so the HTTP API and a background
    retraining job can use the same ``Database`` object safely.
    """

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        self._initialised = False

        # Repositories are cheap wrappers; they resolve the connection lazily.
        self.predictions = PredictionRepository(self)
        self.outcomes = OutcomeRepository(self)
        self.experiences = ExperienceRepository(self)
        self.models = ModelRepository(self)
        self.backtests = BacktestRepository(self)
        self.datasets = DatasetRepository(self)
        self.events = EventRepository(self)
        self.paper_trades = PaperTradeRepository(self)

    # -- connection management --------------------------------------------
    def connection(self) -> sqlite3.Connection:
        """Return this thread's connection, creating it on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA busy_timeout=30000")
            except sqlite3.DatabaseError:
                pass  # :memory: or exotic filesystems may refuse WAL
            self._local.conn = conn
        return conn

    class _CursorContext:
        """Commits on success, rolls back on failure, always closes the cursor."""

        def __init__(self, db: "Database") -> None:
            self._db = db
            self._cur: Optional[sqlite3.Cursor] = None

        def __enter__(self) -> sqlite3.Cursor:
            self._cur = self._db.connection().cursor()
            return self._cur

        def __exit__(self, exc_type, exc, tb) -> bool:
            try:
                if self._cur is not None:
                    self._cur.close()
            finally:
                conn = self._db.connection()
                if exc_type is None:
                    conn.commit()
                else:
                    conn.rollback()
            return False

    def cursor(self) -> "Database._CursorContext":
        """Context manager yielding a cursor with automatic commit/rollback."""
        return Database._CursorContext(self)

    # -- lifecycle ---------------------------------------------------------
    def initialize(self) -> "Database":
        """Create tables and indexes.  Idempotent."""
        with self._lock:
            conn = self.connection()
            for statement in _SCHEMA:
                conn.execute(statement)
            conn.commit()
            self._initialised = True
        return self

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def reset(self) -> None:
        """Drop and recreate all tables.  Used by tests only."""
        with self._lock:
            conn = self.connection()
            for table in (
                "predictions", "outcomes", "experiences", "models",
                "model_history", "backtests", "datasets", "events", "paper_trades",
            ):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()
        self.initialize()

    # -- convenience -------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Row counts used by the dashboard."""
        return {
            "predictions": self.predictions.count(),
            "unresolved": self.predictions.count(resolved=0),
            "outcomes": self.outcomes.count(),
            "experiences": self.experiences.count(),
            "models": self.models.count(),
            "champions": self.models.count(role="champion"),
            "challengers": self.models.count(role="challenger"),
            "backtests": self.backtests.count(),
            "paper_trades": self.paper_trades.count(),
            "events": self.events.count(),
        }

    def __enter__(self) -> "Database":
        return self.initialize()

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
