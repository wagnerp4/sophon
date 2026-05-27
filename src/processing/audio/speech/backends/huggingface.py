from __future__ import annotations

import numpy as np

# TODO(tts): Additional Hugging Face speech engines can live here next to Qwen CustomVoice.

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

    def synthesize(self, text: str, speaker: str, language: str, instruct: str | None) -> tuple[np.ndarray, int]:
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
        wave = wavs[0]
        return np.asarray(wave), int(sr)
