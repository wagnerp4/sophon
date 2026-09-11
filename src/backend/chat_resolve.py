from __future__ import annotations

import os
from typing import Literal

ChatBackendId = Literal["hf", "ollama", "lmstudio"]
_CHAT_BACKEND_IDS: tuple[ChatBackendId, ...] = ("hf", "ollama", "lmstudio")


def chat_backend_ids() -> tuple[ChatBackendId, ...]:
    return _CHAT_BACKEND_IDS


def is_server_backend(backend_id: str) -> bool:
    return backend_id in ("ollama", "lmstudio")


def normalize_chat_backend_token(token: str) -> ChatBackendId | None:
    raw = token.strip().lower().replace("_", "-")
    if raw in ("hf", "huggingface", "transformers"):
        return "hf"
    if raw in ("ollama",):
        return "ollama"
    if raw in ("lmstudio", "lm-studio", "lms"):
        return "lmstudio"
    return None


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


def list_server_models(backend_id: ChatBackendId) -> list[str]:
    if backend_id == "ollama":
        from backend.ollama.backend import list_model_names

        return list_model_names()
    if backend_id == "lmstudio":
        from backend.lmstudio import backend as lms

        return lms.list_model_names()
    return []


def server_has_models(backend_id: ChatBackendId) -> bool:
    try:
        return len(list_server_models(backend_id)) > 0
    except Exception:
        return False


def resolve_chat_backend(explicit: str | None = None) -> ChatBackendId:
    raw = (explicit if explicit is not None else os.environ.get("ORODRUIN_CHAT_BACKEND", "auto")).strip().lower()
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
        raise ValueError(f"unknown chat backend {raw!r} (use auto|ollama|lmstudio|hf)")
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
        from backend.ollama.backend import chat_complete

        text = chat_complete(
            model,
            messages,
            max_new_tokens=max_new_tokens,
        )
        from backend.openai_compat import ChatCompletionResult

        return ChatCompletionResult(text=text, tool_calls=[])
    if backend_id == "lmstudio":
        from backend.lmstudio import backend as lms

        return lms.chat_complete(
            model,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
        )
    raise ValueError(f"not a server backend: {backend_id!r}")
