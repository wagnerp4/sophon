from __future__ import annotations

import os
from collections import deque
from typing import Callable

import numpy as np

StatusFn = Callable[[str], None]


def listen_seconds_default() -> float:
    raw = os.environ.get("ORODRUIN_SST_LISTEN_S") or os.environ.get("ORODRUIN_LISTEN_SECONDS", "30")
    token = raw.strip()
    try:
        value = float(token) if token else 30.0
    except ValueError:
        value = 30.0
    return max(1.0, min(value, 120.0))


def _input_device() -> int | str | None:
    raw = os.environ.get("ORODRUIN_SST_MIC", "").strip()
    if raw == "":
        return None
    if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
        return int(raw)
    return raw


def _vad_rms() -> float:
    raw = os.environ.get("ORODRUIN_SST_VAD_RMS", "0.012").strip()
    try:
        value = float(raw) if raw else 0.012
    except ValueError:
        value = 0.012
    return max(0.001, min(value, 0.2))


def _capture_failure(exc: BaseException) -> RuntimeError:
    hint = " Use Windows native orodruin-cli for WASAPI input, not WSL."
    try:
        from utils.device.platform import is_wsl

        if not is_wsl() and os.name != "posix":
            hint = " Check the default input device, ORODRUIN_SST_MIC, and microphone permissions."
        elif not is_wsl():
            hint = " Check the default input device and ORODRUIN_SST_MIC."
    except Exception:
        pass
    return RuntimeError(f"microphone capture failed ({exc}).{hint}")


def record_push_to_talk(
    *,
    sample_rate: int = 16000,
    max_seconds: float = 30.0,
    on_status: StatusFn | None = None,
) -> tuple[np.ndarray, int]:
    # TODO(sst): hold-to-talk keydown/keyup in the TUI instead of energy-VAD-only stop.
    # TODO(sst): optional Silero VAD when energy gating misfires on noisy mics.
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError("sounddevice unavailable - install with: uv sync --extra sst") from exc

    rate = max(8000, int(sample_rate))
    limit_s = max(1.0, min(float(max_seconds), 120.0))
    chunk_s = 0.1
    frames = max(1, int(round(rate * chunk_s)))
    max_chunks = max(1, int(round(limit_s / chunk_s)))
    start_need = 3
    stop_need = 9
    min_speech_chunks = 4
    preroll_n = 4
    threshold = _vad_rms()
    device = _input_device()
    kwargs: dict[str, object] = {
        "samplerate": rate,
        "channels": 1,
        "dtype": "float32",
        "blocksize": frames,
    }
    if device is not None:
        kwargs["device"] = device
    if on_status is not None:
        on_status("listening")
    preroll: deque[np.ndarray] = deque(maxlen=preroll_n)
    speech: list[np.ndarray] = []
    loud_run = 0
    quiet_run = 0
    started = False
    try:
        with sd.InputStream(**kwargs) as stream:
            for _ in range(max_chunks):
                block, _overflowed = stream.read(frames)
                mono = np.asarray(block, dtype=np.float32).reshape(-1)
                if mono.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(np.square(mono))))
                if not started:
                    preroll.append(mono)
                    if rms >= threshold:
                        loud_run += 1
                    else:
                        loud_run = 0
                    if loud_run >= start_need:
                        started = True
                        speech.extend(preroll)
                        quiet_run = 0
                        if on_status is not None:
                            on_status("speech")
                    continue
                speech.append(mono)
                if rms < threshold:
                    quiet_run += 1
                else:
                    quiet_run = 0
                if len(speech) >= min_speech_chunks and quiet_run >= stop_need:
                    break
    except Exception as exc:
        raise _capture_failure(exc) from exc
    if not speech:
        raise RuntimeError("no speech detected")
    wave = np.concatenate(speech, axis=0).astype(np.float32)
    return wave, rate
