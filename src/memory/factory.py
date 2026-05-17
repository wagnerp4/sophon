from __future__ import annotations

import os
from pathlib import Path

from .protocols import MemoryStore
from .sqlite_memory import SqliteMemoryStore


def open_memory_store(path: str | Path | None) -> MemoryStore | None:
    """Resolve --memory-db path. Falls back to MITHRIL_MEMORY_DB env. Returns None if neither is set."""
    resolved = ""
    if path is not None:
        resolved = str(path).strip()
    if not resolved:
        resolved = os.environ.get("MITHRIL_MEMORY_DB", "").strip()
    if not resolved:
        return None
    return SqliteMemoryStore(resolved)
