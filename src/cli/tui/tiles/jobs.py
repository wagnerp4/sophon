from __future__ import annotations

import time
from datetime import date
from typing import Any

from cli.tui.tiles.base import BaseTile, TileState, wrap_fetch


def _fetch_jobs() -> TileState:
    from integrations.jobs import paths
    from integrations.jobs.criterion import load_criterion
    from integrations.jobs.feeds import fetch_roles
    from integrations.jobs.match import match_jobs
    from integrations.jobs.store import VaultStore

    store = VaultStore()
    check = str(load_criterion(store).get("check") or "daily")
    if store.exists(paths.DIGEST) and _digest_stale(store.read(paths.DIGEST), check):
        fetch_roles(store, pause_s=0.4)
        match_jobs(store)
    if not store.exists(paths.DIGEST):
        return TileState(
            ok=True,
            text="jobs\n(no digest)\n/jobs match",
            fetched_at=time.time(),
        )
    text = store.read(paths.DIGEST)
    best, new_rows = _read_digest(text)
    lines = ["jobs", "check: " + check]
    if best is None:
        lines.append("(empty)")
    else:
        lines.append("best: " + best["legal_name"] + "  " + best["score"])
        if best["role"]:
            lines.append("  " + best["role"])
    lines.append("new: " + str(len(new_rows)))
    for row in new_rows[:3]:
        lines.append("  " + row["legal_name"] + " — " + row["title"])
    return TileState(ok=True, text="\n".join(lines), fetched_at=time.time())


def _digest_stale(text: str, check: str) -> bool:
    limit = 7 if check == "weekly" else 1
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("Updated:"):
            continue
        stamp = line.split(":", 1)[1].strip()
        try:
            day = date.fromisoformat(stamp)
        except ValueError:
            return True
        return (date.today() - day).days >= limit
    return True


def _read_digest(text: str) -> tuple[dict[str, str] | None, list[dict[str, str]]]:
    best: dict[str, str] | None = None
    role = ""
    section = ""
    new_rows: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            if current.get("title"):
                new_rows.append(current)
                current = {}
            section = line[3:].strip().lower()
            continue
        if section == "top" and best is None and line[:1].isdigit() and ". " in line:
            body = line.split(". ", 1)[1]
            legal, score = body, ""
            if " — " in body:
                legal, rest = body.split(" — ", 1)
                score = rest.split(" — ", 1)[0].strip()
            best = {"legal_name": legal.strip(), "score": score, "role": ""}
            role = ""
            continue
        if section == "top" and best is not None and line.startswith("- role:"):
            role = line.split(":", 1)[1].strip()
            best["role"] = role
            continue
        if section != "new":
            continue
        if line.startswith("- company:"):
            if current.get("title"):
                new_rows.append(current)
            current = {"legal_name": line.split(":", 1)[1].strip(), "title": ""}
            continue
        if line.startswith("title:") and current:
            current["title"] = line.split(":", 1)[1].strip()
    if current.get("title"):
        new_rows.append(current)
    return best, new_rows


class JobsTile(BaseTile):
    tile_id = "jobs"
    refresh_s = 120.0

    def fetch(self, app: Any) -> TileState:
        _ = app
        return wrap_fetch(self.tile_id, "jobs", _fetch_jobs)
