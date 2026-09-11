from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .protocols import MemoryStore
from .types import MemoryScope, StoredTurn

SCHEMA_VERSION = 1

_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT,
        created_at REAL NOT NULL,
        last_seen_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        user_id TEXT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at REAL NOT NULL,
        meta_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, rowid);",
    "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, rowid);",
)


class SqliteMemoryStore(MemoryStore):
    """SQLite-backed chat memory using stdlib sqlite3. File path may be ':memory:' for ephemeral use."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._path = str(Path(self._path).expanduser())
        self._conn = sqlite3.connect(self._path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        try:
            for stmt in _SCHEMA_SQL:
                cur.execute(stmt)
            cur.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?);",
                ("schema_version", str(SCHEMA_VERSION)),
            )
        finally:
            cur.close()

    def _ensure_session(self, scope: MemoryScope, now: float) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO sessions(session_id, user_id, created_at, last_seen_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET last_seen_at = excluded.last_seen_at, "
                "user_id = COALESCE(sessions.user_id, excluded.user_id);",
                (scope.session_id, scope.user_id, now, now),
            )
        finally:
            cur.close()

    def append_turn(
        self,
        scope: MemoryScope,
        role: str,
        content: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> StoredTurn:
        now = time.time()
        self._ensure_session(scope, now)
        meta_payload = json.dumps(meta or {}, ensure_ascii=False, sort_keys=True)
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO messages(session_id, user_id, role, content, created_at, meta_json) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                (scope.session_id, scope.user_id, role, content, now, meta_payload),
            )
            rowid = int(cur.lastrowid) if cur.lastrowid is not None else None
        finally:
            cur.close()
        return StoredTurn(
            role=role,
            content=content,
            created_at=now,
            session_id=scope.session_id,
            user_id=scope.user_id,
            meta=dict(meta or {}),
            rowid=rowid,
        )

    def load_recent_turns(self, scope: MemoryScope, limit: int) -> list[StoredTurn]:
        if limit <= 0:
            return []
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT rowid, session_id, user_id, role, content, created_at, meta_json "
                "FROM messages WHERE session_id = ? ORDER BY rowid DESC LIMIT ?;",
                (scope.session_id, int(limit)),
            )
            rows = cur.fetchall()
        finally:
            cur.close()
        turns: list[StoredTurn] = []
        for rowid, session_id, user_id, role, content, created_at, meta_json in rows:
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except json.JSONDecodeError:
                meta = {}
            turns.append(
                StoredTurn(
                    role=str(role),
                    content=str(content),
                    created_at=float(created_at),
                    session_id=str(session_id),
                    user_id=user_id,
                    meta=meta if isinstance(meta, dict) else {},
                    rowid=int(rowid),
                )
            )
        turns.reverse()
        return turns

    def clear_scope(self, scope: MemoryScope) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute("DELETE FROM messages WHERE session_id = ?;", (scope.session_id,))
            removed = int(cur.rowcount or 0)
            cur.execute("DELETE FROM sessions WHERE session_id = ?;", (scope.session_id,))
        finally:
            cur.close()
        return removed

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
