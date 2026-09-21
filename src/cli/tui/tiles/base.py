from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from textual.widgets import Static

from utils.device.env_bootstrap import sophon_project_root


@dataclass
class TileState:
    ok: bool
    text: str
    fetched_at: float
    error: str = ""
    stale: bool = False
    payload: dict[str, Any] | None = None

    def age_label(self) -> str:
        age = max(0, int(time.time() - self.fetched_at))
        if age < 60:
            return f"{age}s ago"
        if age < 3600:
            return f"{age // 60}m ago"
        return f"{age // 3600}h ago"


class TilePanel(Static):
    DEFAULT_CSS = """
    TilePanel {
        border: solid $primary;
        padding: 0 1;
        height: 1fr;
        min-height: 6;
    }
    """

    def apply(self, state: TileState) -> None:
        self.update(state.text)


class DashboardTile(Protocol):
    tile_id: str
    refresh_s: float
    span_cols: int

    def fetch(self, app: Any) -> TileState:
        ...

    def render(self, state: TileState) -> str:
        ...

    def panel(self) -> Any:
        ...


class BaseTile:
    tile_id = "tile"
    refresh_s = 60.0
    span_cols = 1

    def panel(self) -> TilePanel:
        panel = TilePanel(id=f"dash-{self.tile_id}")
        if self.span_cols > 1:
            panel.add_class("tile-span-2")
        return panel

    def render(self, state: TileState) -> str:
        return state.text


def dashboard_cache_dir() -> Path:
    path = sophon_project_root() / "data" / "dashboard"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(tile_id: str) -> Path:
    return dashboard_cache_dir() / f"{tile_id}.json"


def load_cached_state(tile_id: str) -> TileState | None:
    path = cache_path(tile_id)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return TileState(
        ok=bool(raw.get("ok")),
        text=str(raw.get("text") or ""),
        fetched_at=float(raw.get("fetched_at") or 0.0),
        error=str(raw.get("error") or ""),
        stale=True,
        payload=raw.get("payload") if isinstance(raw.get("payload"), dict) else None,
    )


def save_cached_state(tile_id: str, state: TileState) -> None:
    path = cache_path(tile_id)
    try:
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    except OSError:
        pass


def http_get_json(url: str, *, timeout_s: float = 12.0, headers: dict[str, str] | None = None) -> Any:
    hdrs = {"User-Agent": "sophon-dashboard/0.1"}
    if headers:
        hdrs.update(headers)
    req = Request(url, headers=hdrs)
    with urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def http_get_text(url: str, *, timeout_s: float = 12.0, headers: dict[str, str] | None = None) -> str:
    hdrs = {"User-Agent": "sophon-dashboard/0.1"}
    if headers:
        hdrs.update(headers)
    req = Request(url, headers=hdrs)
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def http_post_form(
    url: str,
    data: dict[str, str],
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 12.0,
) -> Any:
    from urllib.parse import urlencode

    hdrs = {
        "User-Agent": "sophon-dashboard/0.1",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    if headers:
        hdrs.update(headers)
    body = urlencode(data).encode("utf-8")
    req = Request(url, data=body, headers=hdrs, method="POST")
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def failed_or_cached(tile_id: str, error: str, *, title: str) -> TileState:
    cached = load_cached_state(tile_id)
    if cached is not None and cached.text.strip():
        cached.stale = True
        cached.error = error
        cached.ok = False
        age = cached.age_label()
        body = cached.text
        prefix = f"{title}\n"
        if body.startswith(prefix):
            body = body[len(prefix) :]
        cached.text = f"{title}\n{body}\n[stale {age}] {error}"
        return cached
    return TileState(
        ok=False,
        text=f"{title}\n{error}",
        fetched_at=time.time(),
        error=error,
        stale=False,
    )


def wrap_fetch(tile_id: str, title: str, fn) -> TileState:
    try:
        state = fn()
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return failed_or_cached(tile_id, str(exc), title=title)
    except Exception as exc:
        return failed_or_cached(tile_id, str(exc), title=title)
    save_cached_state(tile_id, state)
    return state
