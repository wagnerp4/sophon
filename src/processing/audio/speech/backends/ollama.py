from __future__ import annotations

import os

import numpy as np


class OllamaSpeechTtsEngine:
    def __init__(self, model_id: str | None) -> None:
        self.model_id = model_id or os.environ.get("ORODRUIN_OLLAMA_TTS_MODEL") or "llama3.2"

    def synthesize(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: str | None,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        _ = (text, speaker, language, instruct, speed)
        raise NotImplementedError(
            "Ollama TTS is not wired yet. "
            "TODO(processing): add HTTP or CLI speech output for the selected Ollama model."
        )
