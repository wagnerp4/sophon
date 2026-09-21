from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from cli.tui.tiles.base import BaseTile, TileState
from utils.device.env_bootstrap import sophon_project_root

_FILETYPE_CACHE_S = 300.0
_FILETYPE_ROOTS_ENV = "SOPHON_SYSTEM_FILETYPE_ROOTS"
_FILETYPE_MAX_FILES = 8_000
_MAX_DISKS = 4
_BAR_WIDTH = 10
_BAR_FILL = "█"
_BAR_EMPTY = "░"

_filetype_cache: tuple[float, dict[str, Any]] | None = None


def _fmt_bytes(value: int | float) -> str:
    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= div:
            return f"{value / div:.1f}{label}"
    return f"{int(value)}B"


def _fmt_pair(used: int | float, total: int | float) -> str:
    return f"{_fmt_bytes(used)}/{_fmt_bytes(total)}"


def _usage_bar(percent: float, *, width: int = _BAR_WIDTH) -> str:
    pct = max(0.0, min(100.0, float(percent)))
    filled = int(round((pct / 100.0) * width))
    filled = max(0, min(width, filled))
    return (_BAR_FILL * filled) + (_BAR_EMPTY * (width - filled))


def collect_disks() -> list[dict[str, Any]]:
    import psutil

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for part in psutil.disk_partitions(all=False):
        mount = str(part.mountpoint or "")
        if not mount or mount in seen:
            continue
        if part.fstype in ("", "squashfs", "overlay", "tmpfs", "devtmpfs"):
            continue
        seen.add(mount)
        try:
            usage = psutil.disk_usage(mount)
        except (PermissionError, FileNotFoundError, OSError):
            continue
        rows.append(
            {
                "mount": mount,
                "device": str(part.device or ""),
                "fstype": str(part.fstype or ""),
                "total_bytes": int(usage.total),
                "used_bytes": int(usage.used),
                "free_bytes": int(usage.free),
                "percent": float(usage.percent),
            }
        )
    rows.sort(key=lambda row: int(row["used_bytes"]), reverse=True)
    return rows[:_MAX_DISKS]


def _filetype_roots() -> list[Path]:
    raw = os.environ.get(_FILETYPE_ROOTS_ENV, "").strip()
    if raw:
        return [Path(part.strip()).expanduser() for part in raw.split(";") if part.strip()]
    return [sophon_project_root()]


def collect_filetypes() -> dict[str, Any]:
    global _filetype_cache
    now = time.time()
    if _filetype_cache is not None and (now - _filetype_cache[0]) < _FILETYPE_CACHE_S:
        return dict(_filetype_cache[1])

    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", "dist", "build"}
    totals: dict[str, int] = {}
    files = 0
    roots = [root for root in _filetype_roots() if root.exists()]
    for root in roots:
        try:
            for path in root.rglob("*"):
                if files >= _FILETYPE_MAX_FILES:
                    break
                if any(part in skip for part in path.parts):
                    continue
                if not path.is_file():
                    continue
                try:
                    size = int(path.stat().st_size)
                except OSError:
                    continue
                ext = path.suffix.lower() or "(none)"
                totals[ext] = totals.get(ext, 0) + size
                files += 1
        except OSError:
            continue
        if files >= _FILETYPE_MAX_FILES:
            break

    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:6]
    payload = {
        "ok": bool(ranked),
        "files_scanned": files,
        "roots": [str(root) for root in roots],
        "top": [{"ext": ext, "bytes": size} for ext, size in ranked],
        "error": "" if ranked else "no files scanned",
    }
    _filetype_cache = (now, payload)
    return dict(payload)


def _format_disk_line(row: dict[str, Any]) -> str:
    pct = float(row.get("percent") or 0.0)
    mount = str(row.get("mount") or "?")
    free = int(row.get("free_bytes") or 0)
    return (
        f"{mount} {pct:3.0f}% {_usage_bar(pct)} "
        f"{_fmt_pair(int(row.get('used_bytes') or 0), int(row.get('total_bytes') or 1))} "
        f"free {_fmt_bytes(free)}"
    )


class StorageTilePanel(Vertical):
    DEFAULT_CSS = """
    StorageTilePanel {
        border: solid $primary;
        padding: 0 1;
        height: 8;
        min-height: 7;
    }
    #storage-title {
        height: 1;
        text-style: bold;
    }
    .storage-line {
        height: 1;
        max-height: 1;
        color: $text-muted;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    #storage-types {
        height: 1;
        max-height: 1;
        color: $text-muted;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._slot_count = _MAX_DISKS

    def compose(self) -> ComposeResult:
        yield Static("storage", id="storage-title")
        for index in range(self._slot_count):
            yield Static("", id=f"storage-line-{index}", classes="storage-line")
        yield Static("", id="storage-types")

    def apply(self, state: TileState) -> None:
        title = self.query_one("#storage-title", Static)
        payload = state.payload or {}
        disks = payload.get("disks") if isinstance(payload.get("disks"), list) else []
        if not state.ok and not disks:
            title.update("storage")
            for index in range(self._slot_count):
                self.query_one(f"#storage-line-{index}", Static).update("")
            self.query_one("#storage-types", Static).update(state.error or "storage n/a")
            return

        title.update(f"storage  ({len(disks)} volumes)")
        for index in range(self._slot_count):
            line = self.query_one(f"#storage-line-{index}", Static)
            if index >= len(disks):
                line.update("")
                continue
            line.update(_format_disk_line(disks[index]))

        types = payload.get("filetypes") if isinstance(payload.get("filetypes"), dict) else {}
        top = types.get("top") if isinstance(types.get("top"), list) else []
        if top:
            bits = [f"{row.get('ext')} {_fmt_bytes(int(row.get('bytes') or 0))}" for row in top[:3]]
            scanned = int(types.get("files_scanned") or 0)
            self.query_one("#storage-types", Static).update(
                f"types ({scanned}): " + " · ".join(bits)
            )
        else:
            self.query_one("#storage-types", Static).update(
                str(types.get("error") or "types: n/a")
            )


class StorageTile(BaseTile):
    tile_id = "storage"
    refresh_s = 8.0

    def panel(self) -> Any:
        return StorageTilePanel(id=f"dash-{self.tile_id}")

    def fetch(self, app: Any) -> TileState:
        try:
            disks = collect_disks()
            filetypes = collect_filetypes()
        except Exception as exc:
            return TileState(
                ok=False,
                text=f"storage\n{exc}",
                fetched_at=time.time(),
                error=str(exc),
            )
        lines = ["storage"]
        for row in disks:
            lines.append(_format_disk_line(row))
        return TileState(
            ok=True,
            text="\n".join(lines),
            fetched_at=time.time(),
            payload={"disks": disks, "filetypes": filetypes},
        )
