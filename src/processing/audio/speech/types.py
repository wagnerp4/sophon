from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TtsBackendId = Literal["pipecat", "hf_qwen_custom_voice", "ollama"]
SttBackendId = Literal["hf_qwen"]

DEFAULT_SST_MODEL = "Qwen/Qwen3-ASR-0.6B"


@dataclass(frozen=True)
class SttResult:
    text: str
    language: str | None = None


def normalize_tts_backend_token(raw: str) -> TtsBackendId | None:
    v = raw.strip().lower().replace("-", "_")
    if v in ("pipecat", "kokoro", "pipecat_kokoro"):
        return "pipecat"
    if v == "hf_qwen_custom_voice":
        return "hf_qwen_custom_voice"
    if v == "ollama":
        return "ollama"
    return None


def tts_backend_help_tokens() -> str:
    return "pipecat | hf-qwen-custom-voice | ollama"


def normalize_stt_backend_token(raw: str) -> SttBackendId | None:
    v = raw.strip().lower().replace("-", "_")
    if v in ("hf_qwen", "qwen", "qwen3_asr", "hf_qwen_asr", "qwen_asr"):
        return "hf_qwen"
    return None


def stt_backend_help_tokens() -> str:
    return "hf-qwen"


def normalize_stt_language(raw: str | None) -> str | None:
    if raw is None:
        return None
    v = raw.strip()
    if v == "" or v.lower() in ("auto", "none", "clear"):
        return None
    table = {
        "en": "English",
        "de": "German",
        "zh": "Chinese",
        "cmn": "Chinese",
        "yue": "Cantonese",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
        "nl": "Dutch",
        "pl": "Polish",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "cs": "Czech",
        "el": "Greek",
        "hu": "Hungarian",
        "ro": "Romanian",
        "tr": "Turkish",
        "hi": "Hindi",
        "th": "Thai",
        "vi": "Vietnamese",
        "id": "Indonesian",
        "ms": "Malay",
        "fil": "Filipino",
        "fa": "Persian",
        "mk": "Macedonian",
    }
    key = v.lower().replace("_", "-")
    if key in table:
        return table[key]
    base = key.split("-", 1)[0]
    if base in table:
        return table[base]
    return v
