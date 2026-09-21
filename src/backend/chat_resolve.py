from __future__ import annotations

import os

from backend.providers import (
    ChatBackendId,
    backend_help_tokens,
    backend_supports_tools,
    chat_backend_ids,
    is_managed_backend,
    is_server_backend,
    managed_backend_ids,
    missing_provider_key_message,
    normalize_chat_backend_token,
    provider_api_key,
    provider_key_hint,
    provider_spec,
    server_target_prefixes,
)

_CHAT_BACKEND_IDS = chat_backend_ids()


def ollama_reachable(*, timeout_s: float = 2.0) -> bool:
    try:
        from backend.ollama.backend import ping_daemon

        ping_daemon(timeout_s=timeout_s)
        return True
    except Exception:
        return False


def lmstudio_reachable(*, timeout_s: float = 2.0) -> bool:
    try:
        from backend.lmstudio import backend as lms

        return bool(lms.ping_daemon(timeout_s=timeout_s))
    except Exception:
        return False


def openai_reachable(*, timeout_s: float = 2.0) -> bool:
    if not provider_api_key("openai"):
        return False
    try:
        from backend.openai import backend as openai_api

        return bool(openai_api.ping_daemon(timeout_s=timeout_s))
    except Exception:
        return False


def anthropic_reachable(*, timeout_s: float = 2.0) -> bool:
    if not provider_api_key("anthropic"):
        return False
    try:
        from backend.anthropic import backend as anthropic_api

        return bool(anthropic_api.ping_daemon(timeout_s=timeout_s))
    except Exception:
        return False


def google_reachable(*, timeout_s: float = 2.0) -> bool:
    if not provider_api_key("google"):
        return False
    try:
        from backend.google import backend as gemini_api

        return bool(gemini_api.ping_daemon(timeout_s=timeout_s))
    except Exception:
        return False


def server_backend_reachable(backend_id: str, *, timeout_s: float = 2.0) -> bool:
    token = str(backend_id).strip().lower()
    if token == "ollama":
        return ollama_reachable(timeout_s=timeout_s)
    if token == "lmstudio":
        return lmstudio_reachable(timeout_s=timeout_s)
    if token == "openai":
        return openai_reachable(timeout_s=timeout_s)
    if token == "anthropic":
        return anthropic_reachable(timeout_s=timeout_s)
    if token == "google":
        return google_reachable(timeout_s=timeout_s)
    return False


def _load_server_adapter(backend_id: ChatBackendId):
    if backend_id == "ollama":
        from backend.ollama import backend as mod

        return mod
    if backend_id == "lmstudio":
        from backend.lmstudio import backend as mod

        return mod
    if backend_id == "openai":
        from backend.openai import backend as mod

        return mod
    if backend_id == "anthropic":
        from backend.anthropic import backend as mod

        return mod
    if backend_id == "google":
        from backend.google import backend as mod

        return mod
    raise ValueError(f"not a server backend: {backend_id!r}")


def list_server_models(backend_id: ChatBackendId, *, refresh: bool = False) -> list[str]:
    from backend.model_index import load_provider_snapshot, memory_get, memory_set, save_provider_models

    token = str(backend_id).strip().lower()
    if token == "hf":
        return []
    if not refresh:
        cached = memory_get(token)
        if cached is not None:
            return cached
        if is_managed_backend(token):
            names, fetched_at = load_provider_snapshot(token)
            if fetched_at is not None:
                memory_set(token, names)
                return names
            return []
    adapter = _load_server_adapter(token)
    names = list(adapter.list_model_names())
    memory_set(token, names)
    if is_managed_backend(token) and refresh:
        save_provider_models(token, names)
    return names


def sync_managed_model_index(
    backends: tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    from backend.model_index import memory_clear, save_provider_models

    selected = backends or managed_backend_ids()
    out: dict[str, list[str]] = {}
    for backend in selected:
        token = str(backend).strip().lower()
        if not is_managed_backend(token):
            continue
        memory_clear(token)
        if not provider_api_key(token):
            out[token] = []
            continue
        names = list(_load_server_adapter(token).list_model_names(timeout_s=30.0))
        save_provider_models(token, names)
        out[token] = names
    return out


def server_has_models(backend_id: ChatBackendId) -> bool:
    try:
        return len(list_server_models(backend_id)) > 0
    except Exception:
        return False


def resolve_chat_backend(explicit: str | None = None) -> ChatBackendId:
    raw = (explicit if explicit is not None else os.environ.get("SOPHON_CHAT_BACKEND", "auto")).strip().lower()
    if not raw:
        raw = "auto"
    if raw == "auto":
        if ollama_reachable() and server_has_models("ollama"):
            return "ollama"
        if lmstudio_reachable() and server_has_models("lmstudio"):
            return "lmstudio"
        return "hf"
    normalized = normalize_chat_backend_token(raw)
    if normalized is None:
        raise ValueError(f"unknown chat backend {raw!r} (use {backend_help_tokens()})")
    return normalized


def server_chat_complete(
    backend_id: ChatBackendId,
    *,
    model: str,
    messages: list[dict[str, str]] | list[dict],
    max_new_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
):
    if backend_id == "ollama":
        from backend.ollama.backend import chat_complete_result

        return chat_complete_result(
            model,
            messages,
            max_new_tokens=max_new_tokens,
            tools=tools,
        )
    adapter = _load_server_adapter(backend_id)
    return adapter.chat_complete(
        model,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tools=tools,
        tool_choice=tool_choice,
    )


__all__ = [
    "ChatBackendId",
    "anthropic_reachable",
    "backend_help_tokens",
    "backend_supports_tools",
    "chat_backend_ids",
    "google_reachable",
    "is_managed_backend",
    "is_server_backend",
    "list_server_models",
    "lmstudio_reachable",
    "managed_backend_ids",
    "normalize_chat_backend_token",
    "ollama_reachable",
    "openai_reachable",
    "provider_api_key",
    "provider_key_hint",
    "provider_spec",
    "resolve_chat_backend",
    "server_backend_reachable",
    "server_chat_complete",
    "server_has_models",
    "server_target_prefixes",
    "sync_managed_model_index",
]
