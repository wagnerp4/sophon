from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryScope:
    session_id: str
    user_id: str | None = None


@dataclass
class StoredTurn:
    role: str
    content: str
    created_at: float
    session_id: str
    user_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    rowid: int | None = None
