from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

EmitFn = Callable[[str], None]


def plain_text_for_tts(text: str) -> str:
    t = re.sub(r"```[\s\S]*?```", " ", text)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    return " ".join(t.split())


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip(), True


class LazyQwenCustomVoiceTts:
    # TODO(tts): Fall back to chunked synthesis when a single forward exceeds model limits.

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


def play_wav_file(path: Path) -> None:
    p = str(path)
    sysname = platform.system().lower()
    if sysname == "darwin":
        subprocess.run(["afplay", p], check=False)
        return
    if sysname == "windows":
        import winsound

        winsound.PlaySound(p, winsound.SND_FILENAME)
        return
    for bin_name in ("paplay", "aplay"):
        exe = shutil.which(bin_name)
        if exe:
            subprocess.run([exe, p], check=False)
            return
    ffplay = shutil.which("ffplay")
    if ffplay:
        subprocess.run(
            [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", p],
            check=False,
        )


def speak_custom_voice_blocking(
    *,
    engine: LazyQwenCustomVoiceTts,
    text: str,
    speaker: str,
    language: str,
    instruct: str | None,
    max_chars: int,
    emit: EmitFn,
    plain_text: bool,
) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        emit(f"(tts skipped: soundfile not installed: {exc})")
        return

    raw = plain_text_for_tts(text) if plain_text else text
    trimmed, truncated = _truncate(raw.strip(), max_chars)
    if not trimmed:
        emit("(tts skipped: empty text)")
        return
    if truncated:
        emit(f"(tts: truncated to {max_chars} characters)")

    path: Path | None = None
    try:
        wav, sr = engine.synthesize(trimmed, speaker=speaker, language=language, instruct=instruct)
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


def register_tts_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Synthesize assistant output with Qwen3-TTS CustomVoice and play audio (requires mithril[tts]).",
    )
    parser.add_argument(
        "--tts-model",
        default=os.environ.get("MITHRIL_TTS_MODEL") or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="Hugging Face repo id for CustomVoice checkpoint.",
    )
    parser.add_argument(
        "--tts-speaker",
        default=os.environ.get("MITHRIL_TTS_SPEAKER") or "Ryan",
        help="Speaker name passed to generate_custom_voice.",
    )
    parser.add_argument(
        "--tts-language",
        default=os.environ.get("MITHRIL_TTS_LANGUAGE") or "English",
        help="Language label passed to generate_custom_voice.",
    )
    parser.add_argument(
        "--tts-instruct",
        default=os.environ.get("MITHRIL_TTS_INSTRUCT") or None,
        help="Optional style instruct string.",
    )
    parser.add_argument(
        "--tts-max-chars",
        type=int,
        default=int(os.environ.get("MITHRIL_TTS_MAX_CHARS") or "8000"),
        help="Maximum characters sent to TTS per reply.",
    )
    parser.add_argument(
        "--tts-device",
        default=os.environ.get("MITHRIL_TTS_DEVICE") or None,
        help="Torch device_map target e.g. cuda:0 or cpu. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--tts-raw-output",
        action="store_true",
        help="Send raw assistant text to TTS without stripping fenced code blocks.",
    )


def play_tts_from_args(args: argparse.Namespace, text: str, emit: EmitFn) -> None:
    if not bool(getattr(args, "tts", False)):
        return
    instruct = getattr(args, "tts_instruct", None)
    if isinstance(instruct, str) and instruct.strip() == "":
        instruct = None
    device = getattr(args, "tts_device", None)
    device_s = device.strip() if isinstance(device, str) and device.strip() else None
    engine = LazyQwenCustomVoiceTts(str(getattr(args, "tts_model")), device_s)
    speak_custom_voice_blocking(
        engine=engine,
        text=text,
        speaker=str(getattr(args, "tts_speaker")),
        language=str(getattr(args, "tts_language")),
        instruct=instruct if isinstance(instruct, str) else None,
        max_chars=max(int(getattr(args, "tts_max_chars")), 1),
        emit=emit,
        plain_text=not bool(getattr(args, "tts_raw_output", False)),
    )
