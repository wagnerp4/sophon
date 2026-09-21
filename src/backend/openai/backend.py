from __future__ import annotations

import os
from typing import Any

from backend.openai_compat import (
    chat_completions,
    list_openai_models,
    openai_compat_base_url,
)
from backend.providers import first_env, provider_spec

_DEFAULT_BASE = "https://api.openai.com/v1"


def openai_base_url() -> str:
    spec = provider_spec("openai")
    default = spec.default_base_url if spec is not None else _DEFAULT_BASE
    return openai_compat_base_url(
        ("SOPHON_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
        default,
    )


def openai_api_key() -> str | None:
    spec = provider_spec("openai")
    keys = spec.env_keys if spec is not None else ("SOPHON_OPENAI_API_KEY", "OPENAI_API_KEY")
    return first_env(keys)


def key_configured() -> bool:
    return bool(openai_api_key())


def uses_max_completion_tokens(model: str) -> bool:
    token = str(model or "").strip().lower()
    return token.startswith(("o1", "o3", "o4", "gpt-5"))


def is_openai_chat_model(model_id: str) -> bool:
    token = str(model_id or "").strip().lower()
    if not token:
        return False
    blocked = (
        "whisper",
        "tts",
        "dall-e",
        "dalle",
        "embedding",
        "moderation",
        "transcribe",
        "audio",
        "realtime",
        "sora",
        "image",
        "babbage",
        "davinci",
        "ada",
        "search",
    )
    if any(part in token for part in blocked):
        return False
    return token.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))


def ping_daemon(base_url: str | None = None, timeout_s: float = 2.0) -> bool:
    _ = base_url
    _ = timeout_s
    return key_configured()


def list_model_names(base_url: str | None = None, timeout_s: float = 8.0) -> list[str]:
    key = openai_api_key()
    if not key:
        return []
    names = list_openai_models(
        base_url or openai_base_url(),
        api_key=key,
        timeout_s=timeout_s,
    )
    return [name for name in names if is_openai_chat_model(name)]


def chat_complete(
    model: str,
    messages: list[dict[str, str]] | list[dict],
    *,
    max_new_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    base_url: str | None = None,
    timeout_s: float | None = None,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
):
    key = openai_api_key()
    if not key:
        from utils.device.env_bootstrap import dotenv_location_label

        raise RuntimeError(f"Set SOPHON_OPENAI_API_KEY or OPENAI_API_KEY in {dotenv_location_label()}")
    deadline = 600.0
    if timeout_s is not None:
        deadline = float(timeout_s)
    else:
        raw = os.environ.get("SOPHON_OPENAI_TIMEOUT_S", "").strip()
        if raw:
            deadline = float(raw)
    omit_sampling = uses_max_completion_tokens(model)
    kwargs: dict[str, Any] = {
        "base_url": base_url or openai_base_url(),
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "api_key": key,
        "timeout_s": deadline,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_token_field": "max_completion_tokens" if uses_max_completion_tokens(model) else "max_tokens",
    }
    if not omit_sampling:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    try:
        return chat_completions(**kwargs)
    except RuntimeError as exc:
        text = str(exc).lower()
        if omit_sampling or "temperature" not in text:
            raise
        kwargs["temperature"] = None
        kwargs["top_p"] = None
        return chat_completions(**kwargs)
