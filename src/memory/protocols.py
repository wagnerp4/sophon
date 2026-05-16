from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import MemoryScope, StoredTurn


@runtime_checkable
class MemoryStore(Protocol):
    """Persistent chat memory. Implementations may also be in-process (e.g. tests)."""

    def append_turn(
        self,
        scope: MemoryScope,
        role: str,
        content: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> StoredTurn: ...

    def load_recent_turns(self, scope: MemoryScope, limit: int) -> list[StoredTurn]: ...

    def clear_scope(self, scope: MemoryScope) -> int:
        """Drop all turns for scope. Returns number of removed rows."""
        ...

    def close(self) -> None: ...
