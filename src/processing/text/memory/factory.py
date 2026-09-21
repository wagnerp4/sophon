from __future__ import annotations

import os
from pathlib import Path

from .layer import MemoryLayer
from .markdown_store import MarkdownMemoryStore
from .protocols import MemoryStore
from .sqlite_memory import SqliteMemoryStore
from .types import MemoryScope

_DISABLE = frozenset({"0", "off", "false", "no", "none", "disabled"})


def default_memory_db_path() -> Path:
    from utils.device.env_bootstrap import sophon_data_dir, sophon_project_root
    from utils.device.platform import is_windows_mount_path, is_wsl

    root = sophon_project_root()
    if is_wsl() and is_windows_mount_path(root):
        return Path.home() / ".cache" / "sophon" / "memory" / "memory.db"
    return sophon_data_dir() / "memory" / "memory.db"


def resolve_memory_db_path(path: str | Path | None) -> str | None:
    resolved = str(path).strip() if path is not None else ""
    if not resolved:
        resolved = os.environ.get("SOPHON_MEMORY_DB", "").strip()
    if resolved.lower() in _DISABLE:
        return None
    if not resolved:
        return str(default_memory_db_path())
    if resolved == ":memory:":
        return resolved
    target = Path(resolved).expanduser()
    if not target.is_absolute():
        from utils.device.env_bootstrap import sophon_project_root

        target = sophon_project_root() / target
    return str(target)


def open_memory_store(path: str | Path | None) -> MemoryStore | None:
    resolved = resolve_memory_db_path(path)
    if resolved is None:
        return None
    return SqliteMemoryStore(resolved)


def open_memory_layer(
    memory_store: MemoryStore | None,
    scope: MemoryScope | None,
    project_root: Path | None = None,
) -> MemoryLayer | None:
    if memory_store is None or scope is None:
        return None
    if not isinstance(memory_store, SqliteMemoryStore):
        return None
    from utils.device.env_bootstrap import sophon_memory_dir

    markdown = MarkdownMemoryStore(sophon_memory_dir(project_root))
    return MemoryLayer(memory_store, markdown, scope)
