from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

ChatBackendId = Literal[
    "hf", 
    "ollama", 
    "lmstudio", 
    "openai", 
    "anthropic", 
    "google"
]
ProtocolId = Literal[
    "hf_local",
    "ollama_native",
    "openai_compat",
    "anthropic_messages",
    "gemini_openai_compat",
]

_CHAT_BACKEND_IDS: tuple[ChatBackendId, ...] = (
    "hf",
    "ollama",
    "lmstudio",
    "openai",
    "anthropic",
    "google",
)


@dataclass(frozen=True)
class ProviderSpec:
    id: ChatBackendId
    aliases: tuple[str, ...]
    env_keys: tuple[str, ...]
    default_base_url: str
    protocol: ProtocolId
    supports_tools: bool
    is_local: bool
    key_hint: str


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="hf",
        aliases=("hf", "huggingface", "transformers"),
        env_keys=(),
        default_base_url="",
        protocol="hf_local",
        supports_tools=True,
        is_local=True,
        key_hint="",
    ),
    ProviderSpec(
        id="ollama",
        aliases=("ollama",),
        env_keys=(),
        default_base_url="http://127.0.0.1:11434",
        protocol="ollama_native",
        supports_tools=True,
        is_local=True,
        key_hint="",
    ),
    ProviderSpec(
        id="lmstudio",
        aliases=("lmstudio", "lm-studio", "lms"),
        env_keys=("SOPHON_LM_STUDIO_API_KEY", "LM_STUDIO_API_KEY"),
        default_base_url="http://127.0.0.1:1234/v1",
        protocol="openai_compat",
        supports_tools=True,
        is_local=True,
        key_hint="",
    ),
    ProviderSpec(
        id="openai",
        aliases=("openai",),
        env_keys=("SOPHON_OPENAI_API_KEY", "OPENAI_API_KEY"),
        default_base_url="https://api.openai.com/v1",
        protocol="openai_compat",
        supports_tools=True,
        is_local=False,
        key_hint="SOPHON_OPENAI_API_KEY",
    ),
    ProviderSpec(
        id="anthropic",
        aliases=("anthropic", "claude"),
        env_keys=("SOPHON_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        default_base_url="https://api.anthropic.com",
        protocol="anthropic_messages",
        supports_tools=True,
        is_local=False,
        key_hint="SOPHON_ANTHROPIC_API_KEY",
    ),
    ProviderSpec(
        id="google",
        aliases=("google", "gemini", "google-gemini"),
        env_keys=("SOPHON_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        protocol="gemini_openai_compat",
        supports_tools=True,
        is_local=False,
        key_hint="SOPHON_GEMINI_API_KEY",
    ),
)

_BY_ID: dict[str, ProviderSpec] = {spec.id: spec for spec in PROVIDERS}
_BY_ALIAS: dict[str, ProviderSpec] = {}
for _spec in PROVIDERS:
    for _alias in _spec.aliases:
        _BY_ALIAS[_alias] = _spec


def chat_backend_ids() -> tuple[ChatBackendId, ...]:
    return _CHAT_BACKEND_IDS


def managed_backend_ids() -> tuple[ChatBackendId, ...]:
    return ("openai", "anthropic", "google")


def provider_spec(backend_id: str) -> ProviderSpec | None:
    return _BY_ID.get(str(backend_id).strip().lower())


def first_env(keys: tuple[str, ...]) -> str | None:
    from utils.device.env_bootstrap import strip_env_value

    for key in keys:
        raw = strip_env_value(os.environ.get(key, ""))
        if raw:
            return raw
    return None


def provider_api_key(backend_id: str) -> str | None:
    spec = provider_spec(backend_id)
    if spec is None or not spec.env_keys:
        return None
    return first_env(spec.env_keys)


def provider_key_hint(backend_id: str) -> str:
    spec = provider_spec(backend_id)
    if spec is None:
        return "API key"
    return spec.key_hint or (spec.env_keys[0] if spec.env_keys else "API key")


def missing_provider_key_message(backend_id: str) -> str:
    from utils.device.env_bootstrap import dotenv_location_label

    return f"error: set {provider_key_hint(backend_id)} in {dotenv_location_label()}"


def backend_supports_tools(backend_id: str) -> bool:
    spec = provider_spec(backend_id)
    return bool(spec is not None and spec.supports_tools)


def is_server_backend(backend_id: str) -> bool:
    spec = provider_spec(backend_id)
    return spec is not None and spec.id != "hf"


def is_managed_backend(backend_id: str) -> bool:
    return str(backend_id).strip().lower() in managed_backend_ids()


def normalize_chat_backend_token(token: str) -> ChatBackendId | None:
    raw = token.strip().lower().replace("_", "-")
    spec = _BY_ALIAS.get(raw)
    if spec is None:
        return None
    return spec.id


def server_target_prefixes() -> tuple[tuple[str, ChatBackendId], ...]:
    return (
        ("openai:", "openai"),
        ("anthropic:", "anthropic"),
        ("google:", "google"),
        ("gemini:", "google"),
        ("lmstudio:", "lmstudio"),
        ("lms:", "lmstudio"),
        ("ollama:", "ollama"),
    )


def backend_help_tokens() -> str:
    return "auto|ollama|lmstudio|hf|openai|anthropic|google"
