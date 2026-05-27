from __future__ import annotations

from typing import Literal

TtsBackendId = Literal["hf_qwen_custom_voice", "ollama"]


def normalize_tts_backend_token(raw: str) -> TtsBackendId | None:
    v = raw.strip().lower().replace("-", "_")
    if v == "hf_qwen_custom_voice":
        return "hf_qwen_custom_voice"
    if v == "ollama":
        return "ollama"
    return None


def tts_backend_help_tokens() -> str:
    return "hf-qwen-custom-voice | ollama"
