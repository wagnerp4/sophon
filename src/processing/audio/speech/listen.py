from __future__ import annotations

from pathlib import Path
from typing import Callable

from processing.audio.speech.capture import listen_seconds_default, record_push_to_talk
from processing.audio.speech.protocols import SpeechSttEngine
from processing.audio.speech.types import SttResult, normalize_stt_language

EmitFn = Callable[[str], None]


def transcribe_path_blocking(
    *,
    engine: SpeechSttEngine,
    path: Path,
    language: str | None,
    emit: EmitFn,
) -> SttResult:
    if not path.is_file():
        emit(f"(sst skipped: not a file: {path})")
        return SttResult(text="", language=None)
    lang = normalize_stt_language(language)
    try:
        result = engine.transcribe(str(path), lang)
    except ImportError as exc:
        emit(f"(sst skipped: optional deps missing: {exc})")
        return SttResult(text="", language=None)
    except Exception as exc:
        emit(f"(sst failed: {exc})")
        return SttResult(text="", language=None)
    if not result.text.strip():
        emit("(sst: empty transcript)")
        return result
    return result


def transcribe_wave_blocking(
    *,
    engine: SpeechSttEngine,
    wave,
    sample_rate: int,
    language: str | None,
    emit: EmitFn,
) -> SttResult:
    import numpy as np

    lang = normalize_stt_language(language)
    audio = (np.asarray(wave, dtype=np.float32), int(sample_rate))
    try:
        result = engine.transcribe(audio, lang)
    except ImportError as exc:
        emit(f"(sst skipped: optional deps missing: {exc})")
        return SttResult(text="", language=None)
    except Exception as exc:
        emit(f"(sst failed: {exc})")
        return SttResult(text="", language=None)
    if not result.text.strip():
        emit("(sst: empty transcript)")
        return result
    return result


def listen_and_transcribe_blocking(
    *,
    engine: SpeechSttEngine,
    language: str | None,
    emit: EmitFn,
    sample_rate: int = 16000,
    max_seconds: float | None = None,
) -> SttResult:
    seconds = listen_seconds_default() if max_seconds is None else max(1.0, min(float(max_seconds), 120.0))
    emit(f"(sst listening · max {seconds:.0f}s · stop after silence)")
    seen = {"phase": ""}

    def on_status(phase: str) -> None:
        if phase == seen["phase"]:
            return
        seen["phase"] = phase
        if phase == "speech":
            emit("(sst speech detected)")

    try:
        wave, rate = record_push_to_talk(
            sample_rate=sample_rate,
            max_seconds=seconds,
            on_status=on_status,
        )
    except ImportError as exc:
        emit(f"(sst skipped: optional deps missing: {exc})")
        return SttResult(text="", language=None)
    except Exception as exc:
        emit(f"(sst listen failed: {exc})")
        return SttResult(text="", language=None)
    emit("(sst transcribing)")
    return transcribe_wave_blocking(
        engine=engine,
        wave=wave,
        sample_rate=rate,
        language=language,
        emit=emit,
    )
