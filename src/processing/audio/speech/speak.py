from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from processing.audio.speech.playback import play_wav_file
from processing.audio.speech.protocols import SpeechTtsEngine
from processing.audio.speech.text_prep import plain_text_for_tts, truncate_for_spoken_text

EmitFn = Callable[[str], None]


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

    raw = plain_text_for_tts(text) if plain_text else text
    trimmed, truncated = truncate_for_spoken_text(raw.strip(), max_chars)
    if not trimmed:
        emit("(tts skipped: empty text)")
        return
    if truncated:
        emit(f"(tts: truncated to {max_chars} characters)")

    path: Path | None = None
    try:
        wav, sr = engine.synthesize(
            trimmed,
            speaker=speaker,
            language=language,
            instruct=instruct,
            speed=float(speed),
        )
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
