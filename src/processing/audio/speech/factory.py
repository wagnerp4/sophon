from __future__ import annotations

from processing.audio.speech.backends.huggingface import LazyQwenAsr, LazyQwenCustomVoiceTts
from processing.audio.speech.backends.ollama import OllamaSpeechTtsEngine
from processing.audio.speech.backends.pipecat import PipecatKokoroSpeechTts
from processing.audio.speech.protocols import SpeechSttEngine, SpeechTtsEngine
from processing.audio.speech.types import SttBackendId, TtsBackendId


def create_speech_tts_engine(
    backend_id: TtsBackendId,
    *,
    hf_model_id: str,
    hf_device_hint: str | None,
    ollama_tts_model: str | None,
) -> SpeechTtsEngine:
    if backend_id == "pipecat":
        return PipecatKokoroSpeechTts()
    if backend_id == "hf_qwen_custom_voice":
        return LazyQwenCustomVoiceTts(hf_model_id, hf_device_hint)
    if backend_id == "ollama":
        return OllamaSpeechTtsEngine(ollama_tts_model)
    raise KeyError(f"unknown TTS backend {backend_id!r}")


def create_speech_stt_engine(
    backend_id: SttBackendId,
    *,
    hf_model_id: str,
    hf_device_hint: str | None,
    max_new_tokens: int,
) -> SpeechSttEngine:
    if backend_id == "hf_qwen":
        return LazyQwenAsr(hf_model_id, hf_device_hint, max_new_tokens)
    raise KeyError(f"unknown SST backend {backend_id!r}")
