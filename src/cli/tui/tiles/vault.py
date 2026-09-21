from __future__ import annotations

import os
import time
from datetime import date
from typing import Any

from cli.tui.tiles.base import BaseTile, TileState, wrap_fetch


def _limit() -> int:
    raw = os.environ.get("SOPHON_VAULT_LIMIT", "8").strip()
    try:
        return max(1, min(int(raw or "8"), 30))
    except ValueError:
        return 8


def _daily_candidates() -> list[str]:
    today = date.today()
    iso = today.isoformat()
    compact = today.strftime("%Y%m%d")
    custom = os.environ.get("SOPHON_VAULT_DAILY_PATH", "").strip()
    candidates = []
    if custom:
        candidates.append(custom)
    candidates.extend(
        [
            f"Daily/{iso}.md",
            f"daily/{iso}.md",
            f"{iso}.md",
            f"Journal/{iso}.md",
            f"Daily Notes/{iso}.md",
            f"Daily/{compact}.md",
        ]
    )
    return candidates


def _fetch_vault() -> TileState:
    # TODO: true mtime-ordered recent notes if Local REST exposes file metadata
    from integrations.obsidian.client import (
        ObsidianClient,
        obsidian_api_key,
        obsidian_api_url,
        obsidian_tools_enabled,
    )

    if not obsidian_tools_enabled() and not obsidian_api_key():
        raise ValueError("set SOPHON_OBSIDIAN_TOOLS=1 and SOPHON_OBSIDIAN_API_KEY for vault tile")
    client = ObsidianClient(timeout_s=8.0)
    ping = client.ping()
    authenticated = bool(ping.get("authenticated")) if isinstance(ping, dict) else False
    status = "auth ok" if authenticated else "reachable"
    limit = _limit()
    root_entries = client.list_dir("")
    recent = [str(item).rstrip("/") for item in root_entries[:limit]]
    daily_path = ""
    daily_preview = ""
    for path in _daily_candidates():
        try:
            text = client.get_file(path)
        except Exception:
            continue
        daily_path = path
        preview_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
        daily_preview = " | ".join(preview_lines)[:120]
        break
    lines = [
        "vault",
        f"{obsidian_api_url()}  [{status}]",
    ]
    if daily_path:
        lines.append(f"daily: {daily_path}")
        if daily_preview:
            lines.append(f"  {daily_preview}")
    else:
        lines.append("daily: (not found)")
    lines.append("[root]")
    if not recent:
        lines.append("  (empty)")
    else:
        for name in recent:
            lines.append(f"  {name}")
    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=time.time(),
        payload={
            "status": status,
            "daily_path": daily_path,
            "recent": recent,
            "ping": ping if isinstance(ping, dict) else {},
        },
    )


class VaultTile(BaseTile):
    tile_id = "vault"
    refresh_s = 120.0

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "vault", _fetch_vault)
