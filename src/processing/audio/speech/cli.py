from __future__ import annotations

import os
from dataclasses import dataclass

from typing import Callable

from processing.audio.speech.factory import create_speech_tts_engine
from processing.audio.speech.speak import speak_text_blocking
from processing.audio.speech.types import (
    DEFAULT_SST_MODEL,
    SttBackendId,
    TtsBackendId,
    normalize_stt_backend_token,
    normalize_stt_language,
    normalize_tts_backend_token,
    stt_backend_help_tokens,
    tts_backend_help_tokens,
)

EmitFn = Callable[[str], None]


def _default_tts_backend_from_env() -> TtsBackendId:
    raw = os.environ.get("SOPHON_TTS_BACKEND")
    if raw is None or raw.strip() == "":
        return "pipecat"
    out = normalize_tts_backend_token(raw)
    if out is None:
        return "pipecat"
    return out


def _default_tts_speaker_from_env(backend: TtsBackendId) -> str:
    raw = os.environ.get("SOPHON_TTS_SPEAKER")
    if raw is not None and raw.strip() != "":
        return raw.strip()
    if backend == "pipecat":
        return "am_adam"
    return "Ryan"


@dataclass
class TtsCliOptions:
    tts_enabled: bool = False
    tts_backend: TtsBackendId = "pipecat"
    tts_model: str = ""
    tts_ollama_model: str | None = None
    tts_speaker: str = "am_adam"
    tts_language: str = "English"
    tts_instruct: str | None = None
    tts_max_chars: int = 8000
    tts_device: str | None = None
    tts_raw_output: bool = False
    tts_speed: float = 1.0

    @staticmethod
    def defaults_from_env() -> "TtsCliOptions":
        backend = _default_tts_backend_from_env()
        speed_raw = os.environ.get("SOPHON_TTS_SPEED", "1.0").strip()
        try:
            speed = float(speed_raw) if speed_raw else 1.0
        except ValueError:
            speed = 1.0
        speed = max(0.5, min(speed, 2.0))
        return TtsCliOptions(
            tts_enabled=False,
            tts_backend=backend,
            tts_model=os.environ.get("SOPHON_TTS_MODEL") or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            tts_ollama_model=os.environ.get("SOPHON_OLLAMA_TTS_MODEL") or None,
            tts_speaker=_default_tts_speaker_from_env(backend),
            tts_language=os.environ.get("SOPHON_TTS_LANGUAGE") or "English",
            tts_instruct=os.environ.get("SOPHON_TTS_INSTRUCT") or None,
            tts_max_chars=int(os.environ.get("SOPHON_TTS_MAX_CHARS") or "8000"),
            tts_device=os.environ.get("SOPHON_TTS_DEVICE") or None,
            tts_raw_output=False,
            tts_speed=speed,
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
    bid: TtsBackendId = normalized if normalized is not None else "pipecat"
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
        speed=float(getattr(opts, "tts_speed", 1.0) or 1.0),
    )


def _default_stt_backend_from_env() -> SttBackendId:
    raw = os.environ.get("SOPHON_SST_BACKEND")
    if raw is None or raw.strip() == "":
        return "hf_qwen"
    out = normalize_stt_backend_token(raw)
    if out is None:
        return "hf_qwen"
    return out


@dataclass
class SttCliOptions:
    sst_enabled: bool = False
    sst_backend: SttBackendId = "hf_qwen"
    sst_model: str = ""
    sst_language: str | None = None
    sst_device: str | None = None
    sst_max_new_tokens: int = 1024

    @staticmethod
    def defaults_from_env() -> "SttCliOptions":
        enabled_raw = os.environ.get("SOPHON_SST", "").strip().lower()
        enabled = enabled_raw in ("1", "true", "yes", "on")
        max_raw = os.environ.get("SOPHON_SST_MAX_NEW_TOKENS", "1024").strip()
        try:
            max_new = int(max_raw) if max_raw else 1024
        except ValueError:
            max_new = 1024
        lang_raw = os.environ.get("SOPHON_SST_LANGUAGE")
        return SttCliOptions(
            sst_enabled=enabled,
            sst_backend=_default_stt_backend_from_env(),
            sst_model=os.environ.get("SOPHON_SST_MODEL") or DEFAULT_SST_MODEL,
            sst_language=normalize_stt_language(lang_raw),
            sst_device=os.environ.get("SOPHON_SST_DEVICE") or os.environ.get("SOPHON_TTS_DEVICE") or None,
            sst_max_new_tokens=max(max_new, 1),
        )


def format_stt_backend_help() -> str:
    return stt_backend_help_tokens()


def parse_stt_backend_value(value: str) -> SttBackendId | None:
    return normalize_stt_backend_token(value)
