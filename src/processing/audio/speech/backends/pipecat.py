from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


_KOKORO_CACHE = Path.home() / ".cache" / "kokoro-onnx"
_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _map_lang(label: str) -> str:
    raw = label.strip().lower().replace("_", "-")
    table = {
        "en": "en-us",
        "english": "en-us",
        "en-us": "en-us",
        "en-gb": "en-gb",
        "es": "es",
        "spanish": "es",
        "fr": "fr-fr",
        "french": "fr-fr",
        "de": "de",
        "german": "de",
        "it": "it",
        "italian": "it",
        "ja": "ja",
        "japanese": "ja",
        "zh": "cmn",
        "chinese": "cmn",
        "cmn": "cmn",
        "hi": "hi",
        "hindi": "hi",
        "pt": "pt",
        "portuguese": "pt-br",
        "pt-br": "pt-br",
    }
    if raw in table:
        return table[raw]
    base = raw.split("-", 1)[0]
    return table.get(base, "en-us")


def _default_voice(speaker: str) -> str:
    raw = speaker.strip()
    if not raw:
        return "am_adam"
    if re.fullmatch(r"[a-z]{2}_[a-z0-9_]+", raw.lower()):
        return raw.lower()
    return "am_adam"


def _download_if_needed(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _ensure_kokoro() -> Any:
    from kokoro_onnx import Kokoro

    model = _KOKORO_CACHE / "kokoro-v1.0.onnx"
    voices = _KOKORO_CACHE / "voices-v1.0.bin"
    _download_if_needed(_MODEL_URL, model)
    _download_if_needed(_VOICES_URL, voices)
    return Kokoro(str(model), str(voices))


class PipecatKokoroSpeechTts:
    """Local Kokoro ONNX TTS (same stack as pipecat-ai[kokoro], direct synthesize)."""

    def __init__(self) -> None:
        self._kokoro = None
        self._voice = ""
        self._lang = ""

    def _engine(self) -> Any:
        if self._kokoro is None:
            try:
                self._kokoro = _ensure_kokoro()
            except ImportError as exc:
                raise ImportError(
                    "pipecat kokoro unavailable - install with: "
                    "uv sync --extra tts  (needs pipecat-ai[kokoro] + kokoro-onnx)"
                ) from exc
        return self._kokoro

    def synthesize(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: str | None,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        _ = instruct
        voice = _default_voice(speaker)
        lang = _map_lang(language)
        rate = max(0.5, min(float(speed), 2.0))
        samples, sample_rate = self._engine().create(
            text,
            voice=voice,
            lang=lang,
            speed=rate,
        )
        wave = np.asarray(samples, dtype=np.float32)
        return wave, int(sample_rate)
