from __future__ import annotations

from processing.audio.speech.backends.huggingface import LazyQwenCustomVoiceTts
from processing.audio.speech.backends.ollama import OllamaSpeechTtsEngine
from processing.audio.speech.protocols import SpeechTtsEngine
from processing.audio.speech.types import TtsBackendId


def create_speech_tts_engine(
    backend_id: TtsBackendId,
    *,
    hf_model_id: str,
    hf_device_hint: str | None,
    ollama_tts_model: str | None,
) -> SpeechTtsEngine:
    if backend_id == "hf_qwen_custom_voice":
        return LazyQwenCustomVoiceTts(hf_model_id, hf_device_hint)
    if backend_id == "ollama":
        return OllamaSpeechTtsEngine(ollama_tts_model)
    raise KeyError(f"unknown TTS backend {backend_id!r}")
