"""Lightweight local telemetry — async fire-and-forget logging of every
request into a SQLite table for offline analytics.

Captures: timestamp, input_mode, query (truncated), grounding_action,
is_verified, citation_count, retry_count, total_latency_ms. Provides simple
aggregations (top queries, p50/p95 latency, refusal rate) for ops dashboards.

Single-table design — no migrations, no extra dependencies, safe on the
4 GB-VRAM offline laptop deployment.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_unix REAL NOT NULL,
    input_mode TEXT,
    query TEXT,
    grounding_action TEXT,
    is_verified INTEGER,
    citation_count INTEGER,
    retry_count INTEGER,
    total_latency_ms REAL,
    trace_id TEXT
)
"""

_INDEX_DDL = "CREATE INDEX IF NOT EXISTS ix_request_log_ts ON request_log(ts_unix)"


class TelemetryTracker:
    """Thread-safe SQLite-backed request log."""

    def __init__(self, db_path: str | Path, max_query_chars: int = 280):
        self.db_path = str(db_path)
        self.max_query_chars = max_query_chars
        self._lock = threading.Lock()
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(_TABLE_DDL)
            conn.execute(_INDEX_DDL)

    def record(
        self,
        *,
        input_mode: str,
        query: str,
        grounding_action: str,
        is_verified: bool,
        citation_count: int,
        retry_count: int,
        total_latency_ms: float,
        trace_id: str = "",
    ) -> None:
        """Insert one request. Swallows exceptions — telemetry must never
        break a live request."""
        q = (query or "")[: self.max_query_chars]
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    "INSERT INTO request_log "
                    "(ts_unix, input_mode, query, grounding_action, is_verified, "
                    "citation_count, retry_count, total_latency_ms, trace_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        input_mode,
                        q,
                        grounding_action,
                        1 if is_verified else 0,
                        int(citation_count),
                        int(retry_count),
                        float(total_latency_ms),
                        trace_id,
                    ),
                )
        except Exception:
            pass

    # ---- Aggregations for ops dashboards ----

    def summary(self, last_n_hours: float = 24.0) -> dict:
        since = time.time() - last_n_hours * 3600.0
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*), AVG(total_latency_ms), "
                "SUM(CASE WHEN grounding_action='refuse' THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN is_verified=1 THEN 1 ELSE 0 END), "
                "AVG(retry_count) "
                "FROM request_log WHERE ts_unix >= ?",
                (since,),
            )
            count, avg_latency, refusals, verified, avg_retry = cur.fetchone()

            p50 = _percentile(conn, since, 0.5)
            p95 = _percentile(conn, since, 0.95)

        return {
            "window_hours": last_n_hours,
            "request_count": count or 0,
            "avg_latency_ms": round(avg_latency or 0.0, 1),
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "refusal_rate": round((refusals or 0) / count, 3) if count else 0.0,
            "verification_rate": round((verified or 0) / count, 3) if count else 0.0,
            "avg_retry_count": round(avg_retry or 0.0, 2),
        }

    def top_queries(
        self, limit: int = 10, last_n_hours: float = 24.0
    ) -> list[dict]:
        since = time.time() - last_n_hours * 3600.0
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "SELECT query, COUNT(*) as n FROM request_log "
                "WHERE ts_unix >= ? AND query != '' "
                "GROUP BY query ORDER BY n DESC LIMIT ?",
                (since, limit),
            )
            return [{"query": q, "count": n} for q, n in cur.fetchall()]


def _percentile(conn: sqlite3.Connection, since: float, p: float) -> float:
    """SQLite percentile via offset trick. Returns 0.0 on empty set."""
    cur = conn.execute(
        "SELECT COUNT(*) FROM request_log WHERE ts_unix >= ?", (since,)
    )
    n = cur.fetchone()[0] or 0
    if n == 0:
        return 0.0
    offset = max(0, min(n - 1, int(p * (n - 1))))
    cur = conn.execute(
        "SELECT total_latency_ms FROM request_log WHERE ts_unix >= ? "
        "ORDER BY total_latency_ms ASC LIMIT 1 OFFSET ?",
        (since, offset),
    )
    row = cur.fetchone()
    return round(row[0], 1) if row else 0.0


_singleton: Optional[TelemetryTracker] = None
_singleton_lock = threading.Lock()


def get_telemetry(db_path: str | Path) -> TelemetryTracker:
    """Process-wide singleton accessor."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = TelemetryTracker(db_path)
    return _singleton
