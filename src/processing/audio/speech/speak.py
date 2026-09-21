from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from processing.audio.speech.playback import play_wav_file
from processing.audio.speech.protocols import SpeechTtsEngine
from processing.audio.speech.session_store import SessionSpeechStore, SpeechClip, SpeechRole
from processing.audio.speech.text_prep import plain_text_for_tts, truncate_for_spoken_text

EmitFn = Callable[[str], None]


def synthesize_speech_wav(
    *,
    engine: SpeechTtsEngine,
    text: str,
    speaker: str,
    language: str,
    instruct: str | None,
    max_chars: int,
    emit: EmitFn,
    plain_text: bool,
    speed: float = 1.0,
):
    raw = plain_text_for_tts(text) if plain_text else text
    trimmed, truncated = truncate_for_spoken_text(raw.strip(), max_chars)
    if not trimmed:
        emit("(tts skipped: empty text)")
        return None
    if truncated:
        emit(f"(tts: truncated to {max_chars} characters)")
    try:
        wav, sr = engine.synthesize(
            trimmed,
            speaker=speaker,
            language=language,
            instruct=instruct,
            speed=float(speed),
        )
    except ImportError as exc:
        emit(f"(tts skipped: optional deps missing: {exc})")
        return None
    except Exception as exc:
        emit(f"(tts playback failed: {exc})")
        return None
    return wav, sr


def speak_text_blocking(
    *,
    engine: SpeechTtsEngine,
    text: str,
    speaker: str,
    language: str,
    instruct: str | None,
    max_chars: int,
    emit: EmitFn,
    plain_text: bool,
    speed: float = 1.0,
) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        emit(f"(tts skipped: soundfile not installed: {exc})")
        return

    rendered = synthesize_speech_wav(
        engine=engine,
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
        max_chars=max_chars,
        emit=emit,
        plain_text=plain_text,
        speed=speed,
    )
    if rendered is None:
        return
    wav, sr = rendered
    path: Path | None = None
    try:
        fh = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        fh.close()
        path = Path(fh.name)
        sf.write(str(path), wav, sr)
        play_wav_file(path)
    except ImportError as exc:
        emit(f"(tts skipped: optional deps missing: {exc})")
    except Exception as exc:
        emit(f"(tts playback failed: {exc})")
    finally:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def speak_text_to_store(
    *,
    engine: SpeechTtsEngine,
    store: SessionSpeechStore,
    text: str,
    speaker: str,
    language: str,
    instruct: str | None,
    max_chars: int,
    emit: EmitFn,
    plain_text: bool,
    speed: float = 1.0,
    role: SpeechRole = "assistant",
) -> SpeechClip | None:
    rendered = synthesize_speech_wav(
        engine=engine,
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
        max_chars=max_chars,
        emit=emit,
        plain_text=plain_text,
        speed=speed,
    )
    if rendered is None:
        return None
    wav, sr = rendered
    try:
        clip = store.save_wave(wav, sr, role=role, source="tts")
    except Exception as exc:
        emit(f"(speech clip save failed: {exc})")
        return None
    try:
        play_wav_file(clip.path)
    except Exception as exc:
        emit(f"(tts playback failed: {exc})")
        return clip
    return clip
