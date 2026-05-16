from __future__ import annotations

from typing import Literal

BackendId = Literal["hf", "ollama"]

_BACKEND_IDS: tuple[BackendId, ...] = ("hf", "ollama")


def backend_ids() -> tuple[BackendId, ...]:
    return _BACKEND_IDS


def load_backend(backend_id: BackendId):
    if backend_id == "hf":
        from backend.hf import backend

        return backend
    if backend_id == "ollama":
        from backend.ollama import backend

        return backend
    raise KeyError(f"unknown backend {backend_id!r}")


def hf_model_preset_keys() -> list[str]:
    from utils.registry import preset_keys_sorted

    return preset_keys_sorted()


def supported_preset_keys(backend_id: BackendId) -> list[str]:
    if backend_id == "hf":
        return hf_model_preset_keys()
    return []


# TODO(custom-arch): Non-Auto loaders (custom model code) should live under backend/hf/
# (new modules or a plugins/ subpackage) and be selected from here. Ollama-specific
# helpers stay under backend/ollama/.

