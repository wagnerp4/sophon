from __future__ import annotations

from typing import Literal

from backend.chat_resolve import ChatBackendId, chat_backend_ids

BackendId = Literal["hf", "ollama", "lmstudio", "openai", "anthropic", "google"]

_BACKEND_IDS: tuple[BackendId, ...] = (
    "hf",
    "ollama",
    "lmstudio",
    "openai",
    "anthropic",
    "google",
)


def backend_ids() -> tuple[BackendId, ...]:
    return _BACKEND_IDS


def load_backend(backend_id: BackendId):
    if backend_id == "hf":
        from backend.hf import backend

        return backend
    if backend_id == "ollama":
        from backend.ollama import backend

        return backend
    if backend_id == "lmstudio":
        from backend.lmstudio import backend

        return backend
    if backend_id == "openai":
        from backend.openai import backend

        return backend
    if backend_id == "anthropic":
        from backend.anthropic import backend

        return backend
    if backend_id == "google":
        from backend.google import backend

        return backend
    raise KeyError(f"unknown backend {backend_id!r}")


def hf_model_preset_keys() -> list[str]:
    from backend.hf.registry import preset_keys_sorted

    return preset_keys_sorted()


def supported_preset_keys(backend_id: BackendId) -> list[str]:
    if backend_id == "hf":
        return hf_model_preset_keys()
    return []


__all__ = [
    "BackendId",
    "ChatBackendId",
    "backend_ids",
    "chat_backend_ids",
    "hf_model_preset_keys",
    "load_backend",
    "supported_preset_keys",
]
