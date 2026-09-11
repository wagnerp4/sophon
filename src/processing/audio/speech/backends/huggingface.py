from __future__ import annotations

import numpy as np

# TODO(tts): Additional Hugging Face speech engines can live here next to Qwen CustomVoice.
# TODO(sst): Optional Qwen3-ForcedAligner timestamps when /transcribe asks for time_stamps.

# TODO(tts): Fall back to chunked synthesis when a single forward exceeds model limits.


class LazyQwenCustomVoiceTts:
    def __init__(self, model_id: str, device_hint: str | None) -> None:
        self.model_id = model_id
        self.device_hint = device_hint.strip() if isinstance(device_hint, str) and device_hint.strip() else None
        self._model = None

    def _pick_dtype(self):
        import torch

        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _pick_device_map(self) -> str:
        import torch

        if self.device_hint:
            return self.device_hint
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from qwen_tts import Qwen3TTSModel

        kwargs = {
            "device_map": self._pick_device_map(),
            "dtype": self._pick_dtype(),
        }
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                attn_implementation="flash_attention_2",
                **kwargs,
            )
        except Exception:
            self._model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)

    def synthesize(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: str | None,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        self._load()
        assert self._model is not None
        kw: dict[str, object] = {
            "text": text,
            "language": language,
            "speaker": speaker,
        }
        if instruct:
            kw["instruct"] = instruct
        wavs, sr = self._model.generate_custom_voice(**kw)
        wave = np.asarray(wavs[0], dtype=np.float32)
        rate = max(0.5, min(float(speed), 2.0))
        if abs(rate - 1.0) > 1e-3 and wave.size > 1:
            n = max(1, int(round(wave.shape[0] / rate)))
            x_old = np.linspace(0.0, 1.0, num=wave.shape[0], endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
            wave = np.interp(x_new, x_old, wave).astype(np.float32)
        return wave, int(sr)


class LazyQwenAsr:
    def __init__(self, model_id: str, device_hint: str | None, max_new_tokens: int) -> None:
        self.model_id = model_id
        self.device_hint = device_hint.strip() if isinstance(device_hint, str) and device_hint.strip() else None
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self._model = None

    def _pick_dtype(self):
        import torch

        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _pick_device_map(self) -> str:
        import torch

        if self.device_hint:
            return self.device_hint
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ImportError(
                "qwen-asr unavailable - install with: uv sync --extra sst"
            ) from exc

        kwargs = {
            "device_map": self._pick_device_map(),
            "dtype": self._pick_dtype(),
            "max_inference_batch_size": 1,
            "max_new_tokens": self.max_new_tokens,
        }
        try:
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_id,
                attn_implementation="flash_attention_2",
                **kwargs,
            )
        except Exception:
            self._model = Qwen3ASRModel.from_pretrained(self.model_id, **kwargs)

    def transcribe(self, audio, language: str | None):
        from pathlib import Path

        from processing.audio.speech.types import SttResult

        self._load()
        assert self._model is not None
        payload: object
        if isinstance(audio, tuple) and len(audio) == 2:
            payload = (np.asarray(audio[0], dtype=np.float32), int(audio[1]))
        elif isinstance(audio, Path):
            payload = str(audio)
        else:
            payload = str(audio)
        results = self._model.transcribe(audio=payload, language=language)
        if not results:
            return SttResult(text="", language=None)
        first = results[0]
        text = str(getattr(first, "text", "") or "").strip()
        detected = getattr(first, "language", None)
        lang = str(detected).strip() if detected else None
        return SttResult(text=text, language=lang or None)
