from __future__ import annotations

from typing import Protocol

import numpy as np


class SpeechTtsEngine(Protocol):
    def synthesize(self, text: str, speaker: str, language: str, instruct: str | None) -> tuple[np.ndarray, int]:
        ...
