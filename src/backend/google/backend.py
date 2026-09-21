from __future__ import annotations

import os
from typing import Any

from backend.openai_compat import _http_json, chat_completions, openai_compat_base_url
from backend.providers import first_env, provider_spec

_DEFAULT_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_DEFAULT_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"
# TODO: Vertex AI (GCP project + ADC) as a separate google-vertex backend
# TODO: follow nextPageToken when listing Gemini models


def gemini_openai_base_url() -> str:
    spec = provider_spec("google")
    default = spec.default_base_url if spec is not None else _DEFAULT_OPENAI_BASE
    return openai_compat_base_url(
        ("SOPHON_GEMINI_BASE_URL", "GEMINI_BASE_URL"),
        default,
    )


def gemini_native_base_url() -> str:
    raw = os.environ.get("SOPHON_GEMINI_NATIVE_BASE_URL", "").strip()
    if raw:
        return raw.rstrip("/")
    return _DEFAULT_NATIVE_BASE


def gemini_api_key() -> str | None:
    spec = provider_spec("google")
    keys = spec.env_keys if spec is not None else (
        "SOPHON_GEMINI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    )
    return first_env(keys)


def key_configured() -> bool:
    return bool(gemini_api_key())


def ping_daemon(base_url: str | None = None, timeout_s: float = 2.0) -> bool:
    _ = base_url
    _ = timeout_s
    return key_configured()


def _native_model_id(name: str) -> str:
    token = str(name or "").strip()
    if token.startswith("models/"):
        return token[len("models/") :]
    return token


def is_gemini_chat_model(item: dict[str, Any]) -> bool:
    methods = item.get("supportedGenerationMethods")
    if isinstance(methods, list):
        allowed = {str(m) for m in methods}
        if "generateContent" not in allowed:
            return False
    mid = _native_model_id(str(item.get("name") or item.get("id") or ""))
    lower = mid.lower()
    if not lower:
        return False
    if "embedding" in lower or "embed" in lower or "aqa" in lower:
        return False
    return "gemini" in lower or lower.startswith("gemma")


def list_model_names(base_url: str | None = None, timeout_s: float = 8.0) -> list[str]:
    _ = base_url
    key = gemini_api_key()
    if not key:
        return []
    url = f"{gemini_native_base_url()}/models"
    data = _http_json(
        "GET",
        url,
        headers={"x-goog-api-key": key},
        timeout_s=timeout_s,
    )
    items = data.get("models") if isinstance(data, dict) else None
    out: list[str] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            if not is_gemini_chat_model(item):
                continue
            mid = _native_model_id(str(item.get("name") or ""))
            if mid:
                out.append(mid)
    return sorted(set(out))


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
    key = gemini_api_key()
    if not key:
        from utils.device.env_bootstrap import dotenv_location_label

        raise RuntimeError(f"Set SOPHON_GEMINI_API_KEY or GEMINI_API_KEY in {dotenv_location_label()}")
    deadline = 600.0
    if timeout_s is not None:
        deadline = float(timeout_s)
    else:
        raw = os.environ.get("SOPHON_GEMINI_TIMEOUT_S", "").strip()
        if raw:
            deadline = float(raw)
    slug = _native_model_id(model)
    try:
        return chat_completions(
            base_url or gemini_openai_base_url(),
            model=slug,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            api_key=key,
            timeout_s=deadline,
            tools=tools,
            tool_choice=tool_choice,
        )
    except RuntimeError as exc:
        text = str(exc).lower()
        if temperature is None or "temperature" not in text:
            raise
        return chat_completions(
            base_url or gemini_openai_base_url(),
            model=slug,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=None,
            top_p=None,
            api_key=key,
            timeout_s=deadline,
            tools=tools,
            tool_choice=tool_choice,
        )
