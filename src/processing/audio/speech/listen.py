from __future__ import annotations

from pathlib import Path
from typing import Callable

from processing.audio.speech.capture import (
    describe_capture,
    quiet_rms_threshold,
)
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
        emit(f"(sst: empty transcript · file {path.name})")
        return result
    return result


def transcribe_wave_blocking(
    *,
    engine: SpeechSttEngine,
    wave,
    sample_rate: int,
    language: str | None,
    emit: EmitFn,
    peak_rms: float | None = None,
) -> SttResult:
    import numpy as np

    lang = normalize_stt_language(language)
    audio = (np.asarray(wave, dtype=np.float32), int(sample_rate))
    stats = describe_capture(audio[0], audio[1], float(peak_rms or 0.0))
    try:
        result = engine.transcribe(audio, lang)
    except ImportError as exc:
        emit(f"(sst skipped: optional deps missing: {exc})")
        return SttResult(text="", language=None)
    except Exception as exc:
        emit(f"(sst failed: {exc} · captured {stats})")
        return SttResult(text="", language=None)
    if not result.text.strip():
        hint = ""
        if peak_rms is not None and peak_rms < quiet_rms_threshold():
            hint = " Mic level is near silence. Check SOPHON_SST_MIC and the default input device."
        emit(f"(sst: empty transcript · captured {stats}).{hint}")
        return result
    return result
