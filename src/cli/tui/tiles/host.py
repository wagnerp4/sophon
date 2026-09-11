from __future__ import annotations

import shutil
import time
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from cli.tui.tiles.base import BaseTile, TileState
from cli.tui.tiles.hardware import fetch_hardware_state
from utils.device.env_bootstrap import orodruin_project_root


def _fmt_bytes(value: object) -> str:
    if not isinstance(value, int) or value < 0:
        return "unknown"
    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= div:
            return f"{value / div:.2f} {label}"
    return f"{value} B"


def _disk_line(path) -> str:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return "disk: unknown"
    return f"disk={_fmt_bytes(usage.free)} free / {_fmt_bytes(usage.total)}"


def format_host_snapshot(app: Any, *, compact: bool = False) -> str:
    try:
        from utils.device.system_check import collect_system_snapshot

        snapshot = collect_system_snapshot(device="all")
    except Exception as exc:
        return f"host unavailable: {exc}"
    host = snapshot.get("host_memory", {})
    devices = snapshot.get("devices", [])
    device_line = "device: cpu"
    if devices:
        first = devices[0]
        if first.get("kind") == "cuda":
            device_line = (
                f"cuda:{first.get('index')} {first.get('name')} "
                f"free~={_fmt_bytes(first.get('free_bytes'))}"
            )
        else:
            device_line = f"{first.get('kind')}: {first.get('name', first.get('detail', 'available'))}"
    ram_line = (
        f"ram={_fmt_bytes(host.get('host_avail_bytes'))} free / "
        f"{_fmt_bytes(host.get('host_total_bytes'))}"
    )
    win_total = host.get("windows_total_bytes")
    win_avail = host.get("windows_avail_bytes")
    if isinstance(win_total, int) and isinstance(win_avail, int):
        ram_line += f"  win={_fmt_bytes(win_avail)} free / {_fmt_bytes(win_total)}"
    model_line = getattr(app, "model_status_text", "unknown")
    cwd = getattr(app, "cwd", orodruin_project_root())
    root = orodruin_project_root()
    if compact:
        return (
            f"py={snapshot.get('python')} torch={snapshot.get('torch')}  "
            f"{ram_line}  {device_line}\n"
            f"model: {model_line}  cwd: {cwd}"
        )
    load_n = int(getattr(app, "load_progress_n", 0) or 0)
    load_total = int(getattr(app, "load_progress_total", 0) or 0)
    load_phase = str(getattr(app, "load_progress_phase", "") or "")
    load_line = f"load: {load_phase or getattr(app, 'load_status', '')}"
    if load_total > 0:
        load_line += f" ({load_n}/{load_total})"
    return "\n".join(
        [
            f"python={snapshot.get('python')} torch={snapshot.get('torch')}",
            ram_line,
            device_line,
            _disk_line(root),
            f"model: {model_line}",
            load_line,
            f"cwd: {cwd}",
        ]
    )


def _hardware_body(hw: TileState) -> str:
    text = (hw.text or "").strip()
    if text.lower().startswith("hardware"):
        lines = text.splitlines()
        return "\n".join(lines[1:]).strip() or text
    return text


class HostTilePanel(Vertical):
    DEFAULT_CSS = """
    HostTilePanel {
        border: solid $primary;
        padding: 0 1;
        height: 1fr;
        min-height: 12;
    }
    #host-title {
        height: 1;
        text-style: bold;
    }
    #host-body {
        height: auto;
        min-height: 5;
    }
    #host-hw-title {
        height: 1;
        text-style: bold;
        margin-top: 1;
    }
    #host-hw-body {
        height: 1fr;
        min-height: 4;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("host", id="host-title")
        yield Static("", id="host-body")
        yield Static("hardware (asking)", id="host-hw-title")
        yield Static("", id="host-hw-body")

    def apply(self, state: TileState) -> None:
        payload = state.payload or {}
        host_text = str(payload.get("host_text") or state.text)
        hw_text = str(payload.get("hardware_text") or "")
        hw_ok = bool(payload.get("hardware_ok", True))
        self.query_one("#host-body", Static).update(host_text)
        title = self.query_one("#host-hw-title", Static)
        title.update("hardware (asking)" if hw_ok else "hardware [partial]")
        self.query_one("#host-hw-body", Static).update(hw_text or "(no hardware data)")


class HostTile(BaseTile):
    tile_id = "host"
    refresh_s = 30.0

    def panel(self) -> Any:
        return HostTilePanel(id=f"dash-{self.tile_id}")

    def fetch(self, app: Any) -> TileState:
        host_text = format_host_snapshot(app, compact=False)
        hw = fetch_hardware_state()
        hw_body = _hardware_body(hw)
        return TileState(
            ok=True,
            text=f"host\n{host_text}\n\nhardware\n{hw_body}",
            fetched_at=time.time(),
            payload={
                "host_text": host_text,
                "hardware_text": hw_body,
                "hardware_ok": hw.ok and not hw.stale,
            },
            error=hw.error,
        )
