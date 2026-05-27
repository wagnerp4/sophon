from __future__ import annotations

import os
from dataclasses import dataclass

from typing import Callable

from processing.audio.speech.factory import create_speech_tts_engine
from processing.audio.speech.speak import speak_text_blocking
from processing.audio.speech.types import TtsBackendId, normalize_tts_backend_token, tts_backend_help_tokens

EmitFn = Callable[[str], None]


def _default_tts_backend_from_env() -> TtsBackendId:
    raw = os.environ.get("MITHRIL_TTS_BACKEND")
    if raw is None or raw.strip() == "":
        return "hf_qwen_custom_voice"
    out = normalize_tts_backend_token(raw)
    if out is None:
        return "hf_qwen_custom_voice"
    return out


@dataclass
class TtsCliOptions:
    tts_enabled: bool = False
    tts_backend: TtsBackendId = "hf_qwen_custom_voice"
    tts_model: str = ""
    tts_ollama_model: str | None = None
    tts_speaker: str = "Ryan"
    tts_language: str = "English"
    tts_instruct: str | None = None
    tts_max_chars: int = 8000
    tts_device: str | None = None
    tts_raw_output: bool = False

    @staticmethod
    def defaults_from_env() -> "TtsCliOptions":
        return TtsCliOptions(
            tts_enabled=False,
            tts_backend=_default_tts_backend_from_env(),
            tts_model=os.environ.get("MITHRIL_TTS_MODEL") or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            tts_ollama_model=os.environ.get("MITHRIL_OLLAMA_TTS_MODEL") or None,
            tts_speaker=os.environ.get("MITHRIL_TTS_SPEAKER") or "Ryan",
            tts_language=os.environ.get("MITHRIL_TTS_LANGUAGE") or "English",
            tts_instruct=os.environ.get("MITHRIL_TTS_INSTRUCT") or None,
            tts_max_chars=int(os.environ.get("MITHRIL_TTS_MAX_CHARS") or "8000"),
            tts_device=os.environ.get("MITHRIL_TTS_DEVICE") or None,
            tts_raw_output=False,
        )


def format_tts_backend_help() -> str:
    return tts_backend_help_tokens()


def parse_tts_backend_value(value: str) -> TtsBackendId | None:
    return normalize_tts_backend_token(value)


def play_tts_from_options(opts: TtsCliOptions, text: str, emit: EmitFn) -> None:
    if not opts.tts_enabled:
        return
    instruct = opts.tts_instruct
    if isinstance(instruct, str) and instruct.strip() == "":
        instruct = None
    device_s = opts.tts_device.strip() if isinstance(opts.tts_device, str) and opts.tts_device.strip() else None
    bid_raw = opts.tts_backend
    normalized = normalize_tts_backend_token(str(bid_raw))
    bid: TtsBackendId = normalized if normalized is not None else "hf_qwen_custom_voice"
    ollama_model_s = (
        opts.tts_ollama_model.strip()
        if isinstance(opts.tts_ollama_model, str) and opts.tts_ollama_model.strip()
        else None
    )
    engine = create_speech_tts_engine(
        bid,
        hf_model_id=str(opts.tts_model),
        hf_device_hint=device_s,
        ollama_tts_model=ollama_model_s,
    )
    speak_text_blocking(
        engine=engine,
        text=text,
        speaker=str(opts.tts_speaker),
        language=str(opts.tts_language),
        instruct=instruct if isinstance(instruct, str) else None,
        max_chars=max(int(opts.tts_max_chars), 1),
        emit=emit,
        plain_text=not bool(opts.tts_raw_output),
    )
