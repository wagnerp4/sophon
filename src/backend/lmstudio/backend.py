from __future__ import annotations

import os

from backend.openai_compat import (
    chat_completions,
    list_openai_models,
    openai_compat_base_url,
    ping_openai_compat,
)

_DEFAULT_BASE = "http://127.0.0.1:1234/v1"


def lmstudio_base_url() -> str:
    return openai_compat_base_url(
        ("ORODRUIN_LM_STUDIO_HOST", "LM_STUDIO_HOST"),
        _DEFAULT_BASE,
    )


def lmstudio_api_key() -> str | None:
    for key in ("ORODRUIN_LM_STUDIO_API_KEY", "LM_STUDIO_API_KEY"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    return None


def ping_daemon(base_url: str | None = None, timeout_s: float = 2.0) -> bool:
    return ping_openai_compat(
        base_url or lmstudio_base_url(),
        api_key=lmstudio_api_key(),
        timeout_s=timeout_s,
    )


def list_model_names(base_url: str | None = None, timeout_s: float = 8.0) -> list[str]:
    return list_openai_models(
        base_url or lmstudio_base_url(),
        api_key=lmstudio_api_key(),
        timeout_s=timeout_s,
    )


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
    deadline = 600.0
    if timeout_s is not None:
        deadline = float(timeout_s)
    else:
        raw = os.environ.get("ORODRUIN_LM_STUDIO_TIMEOUT_S", "").strip()
        if raw:
            deadline = float(raw)
    result = chat_completions(
        base_url or lmstudio_base_url(),
        model=model,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        api_key=lmstudio_api_key(),
        timeout_s=deadline,
        tools=tools,
        tool_choice=tool_choice,
    )
    return result
