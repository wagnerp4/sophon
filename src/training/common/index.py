from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.common.paths import adapters_root


def adapter_index_path(project_root: Path | None = None) -> Path:
    return adapters_root(project_root) / "index.json"


@dataclass
class AdapterIndexEntry:
    name: str
    preset: str
    dataset: str
    run_id: str
    path: str
    created: str
    last_loss: float | None = None
    steps: int | None = None
    parent: str | None = None
    stage_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preset": self.preset,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "path": self.path,
            "created": self.created,
            "last_loss": self.last_loss,
            "steps": self.steps,
            "parent": self.parent,
            "stage_id": self.stage_id,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> AdapterIndexEntry:
        last_loss = raw.get("last_loss")
        steps = raw.get("steps")
        return cls(
            name=str(raw.get("name") or ""),
            preset=str(raw.get("preset") or ""),
            dataset=str(raw.get("dataset") or ""),
            run_id=str(raw.get("run_id") or ""),
            path=str(raw.get("path") or ""),
            created=str(raw.get("created") or ""),
            last_loss=float(last_loss) if isinstance(last_loss, (int, float)) else None,
            steps=int(steps) if isinstance(steps, int) else None,
            parent=str(raw["parent"]) if raw.get("parent") else None,
            stage_id=str(raw["stage_id"]) if raw.get("stage_id") else None,
        )


@dataclass
class AdapterIndex:
    entries: list[AdapterIndexEntry] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"adapters": [entry.to_json() for entry in self.entries]}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> AdapterIndex:
        rows = raw.get("adapters")
        if not isinstance(rows, list):
            return cls()
        entries: list[AdapterIndexEntry] = []
        for item in rows:
            if isinstance(item, dict):
                entries.append(AdapterIndexEntry.from_json(item))
        return cls(entries=entries)


def load_adapter_index(project_root: Path | None = None) -> AdapterIndex:
    path = adapter_index_path(project_root)
    if not path.is_file():
        return AdapterIndex()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AdapterIndex()
    if not isinstance(raw, dict):
        return AdapterIndex()
    return AdapterIndex.from_json(raw)


def save_adapter_index(index: AdapterIndex, project_root: Path | None = None) -> Path:
    path = adapter_index_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_json(), indent=2), encoding="utf-8")
    return path


def unique_adapter_name(index: AdapterIndex, requested: str) -> str:
    base = requested.strip() or "adapter"
    taken = {entry.name for entry in index.entries}
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def resolve_adapter_entry(
    token: str,
    *,
    project_root: Path | None = None,
) -> AdapterIndexEntry | None:
    raw = token.strip()
    if not raw:
        return None
    index = load_adapter_index(project_root)
    for entry in index.entries:
        if entry.name == raw:
            return entry
    lower = raw.lower()
    ci = [entry for entry in index.entries if entry.name.lower() == lower]
    if len(ci) == 1:
        return ci[0]
    run_hits = [entry for entry in index.entries if entry.run_id == raw]
    if len(run_hits) == 1:
        return run_hits[0]
    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        resolved = str(candidate.resolve())
        for entry in index.entries:
            if entry.path == resolved:
                return entry
        return AdapterIndexEntry(
            name=candidate.name,
            preset="",
            dataset="",
            run_id=candidate.name,
            path=resolved,
            created="",
        )
    return None


def register_adapter_entry(
    entry: AdapterIndexEntry,
    *,
    project_root: Path | None = None,
) -> AdapterIndexEntry:
    index = load_adapter_index(project_root)
    name = unique_adapter_name(index, entry.name)
    stored = AdapterIndexEntry(
        name=name,
        preset=entry.preset,
        dataset=entry.dataset,
        run_id=entry.run_id,
        path=entry.path,
        created=entry.created,
        last_loss=entry.last_loss,
        steps=entry.steps,
        parent=entry.parent,
        stage_id=entry.stage_id,
    )
    index.entries.append(stored)
    save_adapter_index(index, project_root)
    return stored
