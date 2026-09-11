from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from processing.audio.speech.types import SttResult


class SpeechTtsEngine(Protocol):
    def synthesize(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: str | None,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        ...


class SpeechSttEngine(Protocol):
    def transcribe(
        self,
        audio: str | Path | tuple[np.ndarray, int],
        language: str | None,
    ) -> SttResult:
        ...
