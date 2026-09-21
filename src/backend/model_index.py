from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.device.env_bootstrap import sophon_project_root

_INDEX_VERSION = 1
_MEMORY: dict[str, list[str]] = {}
_MANAGED = ("openai", "anthropic", "google")


def managed_index_path() -> Path:
    return sophon_project_root() / "src" / "backend" / "catalogs" / "managed_models.json"


def memory_get(backend_id: str) -> list[str] | None:
    names = _MEMORY.get(str(backend_id).strip().lower())
    if names is None:
        return None
    return list(names)


def memory_set(backend_id: str, names: list[str]) -> None:
    _MEMORY[str(backend_id).strip().lower()] = list(names)


def memory_clear(backend_id: str | None = None) -> None:
    if backend_id is None:
        _MEMORY.clear()
        return
    _MEMORY.pop(str(backend_id).strip().lower(), None)


def _empty_payload() -> dict[str, Any]:
    providers = {
        name: {"fetched_at": None, "models": []}
        for name in _MANAGED
    }
    return {
        "version": _INDEX_VERSION,
        "updated_at": None,
        "providers": providers,
    }


def _read_payload() -> dict[str, Any]:
    path = managed_index_path()
    if not path.is_file():
        return _empty_payload()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_payload()
    if not isinstance(data, dict):
        return _empty_payload()
    providers = data.get("providers")
    if not isinstance(providers, dict):
        data["providers"] = _empty_payload()["providers"]
    return data


def load_provider_snapshot(backend_id: str) -> tuple[list[str], str | None]:
    token = str(backend_id).strip().lower()
    providers = _read_payload().get("providers")
    if not isinstance(providers, dict):
        return [], None
    block = providers.get(token)
    if not isinstance(block, dict):
        return [], None
    fetched = block.get("fetched_at")
    fetched_at = fetched if isinstance(fetched, str) and fetched.strip() else None
    raw = block.get("models")
    names: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
    return names, fetched_at


def save_provider_models(backend_id: str, names: list[str]) -> Path:
    token = str(backend_id).strip().lower()
    if token not in _MANAGED:
        raise ValueError(f"not a managed backend: {token!r}")
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = _read_payload()
    providers = payload.setdefault("providers", {})
    if not isinstance(providers, dict):
        providers = {}
        payload["providers"] = providers
    cleaned = sorted({name.strip() for name in names if isinstance(name, str) and name.strip()})
    providers[token] = {
        "fetched_at": now,
        "count": len(cleaned),
        "models": cleaned,
    }
    payload["version"] = _INDEX_VERSION
    payload["updated_at"] = now
    path = managed_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    memory_set(token, cleaned)
    return path
