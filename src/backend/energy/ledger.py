from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from backend.energy.regime import EnergySession


def ledger_path(session: EnergySession) -> Path:
    override = os.environ.get("SOPHON_ENERGY_DIR", "").strip()
    if override:
        return Path(override) / "spend.jsonl"
    return session.project_root / "data" / "energy" / "spend.jsonl"


def append_spend(session: EnergySession, row: dict) -> None:
    path = ledger_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _rows(session: EnergySession) -> list[dict]:
    path = ledger_path(session)
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _parse_ts(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def sum_today(session: EnergySession, *, now: datetime | None = None) -> float:
    moment = now or datetime.now(timezone.utc)
    total = 0.0
    for row in _rows(session):
        stamp = _parse_ts(row.get("ts"))
        if stamp is None or stamp.date() != moment.date():
            continue
        try:
            total += float(row.get("usd") or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def sum_month(session: EnergySession, *, now: datetime | None = None) -> float:
    moment = now or datetime.now(timezone.utc)
    total = 0.0
    for row in _rows(session):
        stamp = _parse_ts(row.get("ts"))
        if stamp is None or stamp.year != moment.year or stamp.month != moment.month:
            continue
        try:
            total += float(row.get("usd") or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def requests_last_minute(session: EnergySession, *, now: datetime | None = None) -> int:
    moment = now or datetime.now(timezone.utc)
    count = 0
    for row in _rows(session):
        stamp = _parse_ts(row.get("ts"))
        if stamp is None:
            continue
        if (moment - stamp).total_seconds() <= 60:
            count += 1
    return count
