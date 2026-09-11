from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from cli.tui.tiles.base import (
    BaseTile,
    TileState,
    dashboard_cache_dir,
    http_get_json,
    wrap_fetch,
)


def _credentials() -> tuple[str, str] | None:
    client_id = os.environ.get("ORODRUIN_TWITCH_CLIENT_ID", "").strip().strip("\"'")
    client_secret = os.environ.get("ORODRUIN_TWITCH_CLIENT_SECRET", "").strip().strip("\"'")
    if not client_id or not client_secret:
        return None
    return client_id, client_secret


def _limit() -> int:
    raw = os.environ.get("ORODRUIN_TWITCH_LIMIT", "8").strip()
    try:
        return max(1, min(int(raw or "8"), 20))
    except ValueError:
        return 8


def _profile_name() -> str:
    return os.environ.get("ORODRUIN_TWITCH_PROFILE", "default").strip() or "default"


def _mode() -> str:
    raw = os.environ.get("ORODRUIN_TWITCH_MODE", "top").strip().lower() or "top"
    if raw in ("channels", "follow", "following", "user"):
        return "channels"
    return "top"


def _channels() -> list[str]:
    raw = os.environ.get("ORODRUIN_TWITCH_CHANNELS", "").strip()
    return [part.strip().lstrip("@").lower() for part in raw.split(",") if part.strip()]


def _language() -> str:
    return os.environ.get("ORODRUIN_TWITCH_LANGUAGE", "").strip().lower()


def _game_name() -> str:
    return os.environ.get("ORODRUIN_TWITCH_GAME", "").strip()


def _token_cache_path():
    return dashboard_cache_dir() / "twitch_token.json"


def _load_cached_token() -> str | None:
    path = _token_cache_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    token = str(raw.get("access_token") or "")
    expires_at = float(raw.get("expires_at") or 0)
    if not token or time.time() >= expires_at - 60:
        return None
    return token


def _save_cached_token(token: str, expires_in: int) -> None:
    path = _token_cache_path()
    payload = {
        "access_token": token,
        "expires_at": time.time() + max(60, int(expires_in)),
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _fetch_app_token(client_id: str, client_secret: str) -> str:
    cached = _load_cached_token()
    if cached:
        return cached
    query = urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        }
    )
    req = Request(
        f"https://id.twitch.tv/oauth2/token?{query}",
        data=b"",
        headers={"User-Agent": "orodruin-dashboard/0.1"},
        method="POST",
    )
    with urlopen(req, timeout=12.0) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    token = str(data.get("access_token") or "")
    if not token:
        raise ValueError("twitch token response missing access_token")
    _save_cached_token(token, int(data.get("expires_in") or 3600))
    return token


def _helix_get(path: str, *, client_id: str, token: str, params: dict[str, str] | None = None) -> Any:
    url = f"https://api.twitch.tv/helix/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    return http_get_json(
        url,
        timeout_s=12.0,
        headers={
            "Client-ID": client_id,
            "Authorization": f"Bearer {token}",
        },
    )


def _fmt_viewers(n: Any) -> str:
    try:
        value = int(n)
    except (TypeError, ValueError):
        return "?"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _short(text: str, width: int = 42) -> str:
    clean = " ".join(text.split())
    if len(clean) <= width:
        return clean
    return clean[: width - 1] + "…"


def _resolve_game_id(client_id: str, token: str, game: str) -> str | None:
    if not game:
        return None
    if game.isdigit():
        return game
    data = _helix_get(
        "games",
        client_id=client_id,
        token=token,
        params={"name": game},
    )
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    first = rows[0]
    if isinstance(first, dict) and first.get("id"):
        return str(first["id"])
    return None


def _stream_rows(data: Any) -> list[dict[str, Any]]:
    rows_raw = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows_raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in rows_raw:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "user_login": str(item.get("user_login") or item.get("user_name") or ""),
                "user_name": str(item.get("user_name") or item.get("user_login") or ""),
                "title": str(item.get("title") or ""),
                "game_name": str(item.get("game_name") or ""),
                "viewer_count": int(item.get("viewer_count") or 0),
                "language": str(item.get("language") or ""),
            }
        )
    rows.sort(key=lambda row: int(row.get("viewer_count") or 0), reverse=True)
    return rows


def _fetch_top_streams(client_id: str, token: str, limit: int) -> list[dict[str, Any]]:
    params: dict[str, str] = {"first": str(limit)}
    game_id = _resolve_game_id(client_id, token, _game_name())
    if game_id:
        params["game_id"] = game_id
    lang = _language()
    if lang:
        params["language"] = lang
    data = _helix_get("streams", client_id=client_id, token=token, params=params)
    return _stream_rows(data)[:limit]


def _fetch_channel_streams(client_id: str, token: str, channels: list[str], limit: int) -> list[dict[str, Any]]:
    if not channels:
        raise ValueError("ORODRUIN_TWITCH_MODE=channels requires ORODRUIN_TWITCH_CHANNELS")
    rows: list[dict[str, Any]] = []
    chunk_size = 100
    for i in range(0, len(channels), chunk_size):
        chunk = channels[i : i + chunk_size]
        query = "&".join(f"user_login={quote(login)}" for login in chunk)
        url = f"https://api.twitch.tv/helix/streams?{query}"
        data = http_get_json(
            url,
            timeout_s=12.0,
            headers={
                "Client-ID": client_id,
                "Authorization": f"Bearer {token}",
            },
        )
        rows.extend(_stream_rows(data))
    return rows[:limit]


def _fetch_twitch() -> TileState:
    # TODO: optional user OAuth token for Helix followed streams (user:read:follows)
    creds = _credentials()
    if creds is None:
        raise ValueError(
            "set ORODRUIN_TWITCH_CLIENT_ID and ORODRUIN_TWITCH_CLIENT_SECRET "
            "(Twitch developer application)"
        )
    client_id, client_secret = creds
    token = _fetch_app_token(client_id, client_secret)
    limit = _limit()
    mode = _mode()
    profile = _profile_name()
    if mode == "channels":
        rows = _fetch_channel_streams(client_id, token, _channels(), limit)
    else:
        rows = _fetch_top_streams(client_id, token, limit)
    lines = [f"twitch [{profile}/{mode}]"]
    if not rows:
        lines.append("(no live streams)")
    else:
        for row in rows:
            login = row.get("user_login") or row.get("user_name") or "?"
            viewers = _fmt_viewers(row.get("viewer_count"))
            game = _short(str(row.get("game_name") or "-"), 18)
            title = _short(str(row.get("title") or ""), 36)
            lines.append(f"{viewers:>5}  {login:<14}  {game}  {title}")
    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=time.time(),
        payload={
            "profile": profile,
            "mode": mode,
            "streams": rows,
        },
    )


class TwitchTile(BaseTile):
    tile_id = "twitch"
    refresh_s = 60.0
    span_cols = 2

    def panel(self):
        panel = super().panel()
        panel.add_class("tile-span-2")
        return panel

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "twitch", _fetch_twitch)
