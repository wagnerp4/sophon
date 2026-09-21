from __future__ import annotations

from backend.model_index import load_provider_snapshot


def known_managed_teacher_ids(backend_id: str) -> list[str]:
    names, fetched_at = load_provider_snapshot(backend_id)
    if fetched_at is None:
        return []
    return names
