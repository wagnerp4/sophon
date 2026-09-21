from __future__ import annotations

import os
import threading
import time
from typing import Callable

import numpy as np

StatusFn = Callable[[str], None]


def record_max_seconds_default() -> float:
    raw = (
        os.environ.get("SOPHON_SST_MAX_S")
        or os.environ.get("SOPHON_SST_LISTEN_S")
        or os.environ.get("SOPHON_LISTEN_SECONDS", "120")
    )
    token = raw.strip()
    try:
        value = float(token) if token else 120.0
    except ValueError:
        value = 120.0
    return max(2.0, min(value, 300.0))


def listen_seconds_default() -> float:
    return record_max_seconds_default()


def _input_device() -> int | str | None:
    raw = os.environ.get("SOPHON_SST_MIC", "").strip()
    if raw == "":
        return None
    if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
        return int(raw)
    return raw


def quiet_rms_threshold() -> float:
    raw = os.environ.get("SOPHON_SST_VAD_RMS", "0.012").strip()
    try:
        value = float(raw) if raw else 0.012
    except ValueError:
        value = 0.012
    return max(0.001, min(value, 0.2))


def _capture_failure(exc: BaseException) -> RuntimeError:
    hint = " Use Windows native sophon-cli for WASAPI input, not WSL."
    try:
        from utils.device.platform import is_wsl

        if not is_wsl() and os.name != "posix":
            hint = " Check the default input device, SOPHON_SST_MIC, and microphone permissions."
        elif not is_wsl():
            hint = " Check the default input device and SOPHON_SST_MIC."
    except Exception:
        pass
    return RuntimeError(f"microphone capture failed ({exc}).{hint}")


def describe_capture(wave: np.ndarray, sample_rate: int, peak_rms: float) -> str:
    rate = max(int(sample_rate), 1)
    samples = int(np.asarray(wave).size)
    duration = samples / float(rate)
    return f"{duration:.1f}s · {samples} samples · peak RMS {peak_rms:.4f}"


class MicRecorder:
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        max_seconds: float | None = None,
        on_status: StatusFn | None = None,
    ) -> None:
        self._rate = max(8000, int(sample_rate))
        self._max_seconds = (
            record_max_seconds_default()
            if max_seconds is None
            else max(2.0, min(float(max_seconds), 300.0))
        )
        self._on_status = on_status
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._finish_lock = threading.Lock()
        self._finishing = False
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._wave: np.ndarray | None = None
        self._peak_rms = 0.0
        self._hit_max = False
        self._started_at = 0.0
        self._cancelled = False

    @property
    def active(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def finishing(self) -> bool:
        return self._finishing

    @property
    def elapsed_s(self) -> float:
        if self._started_at <= 0.0:
            return 0.0
        end = time.monotonic()
        if self._done.is_set():
            return min(self._max_seconds, max(0.0, end - self._started_at))
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def peak_rms(self) -> float:
        return self._peak_rms

    @property
    def hit_max(self) -> bool:
        return self._hit_max

    @property
    def max_seconds(self) -> float:
        return self._max_seconds

    def take_finish(self) -> bool:
        with self._finish_lock:
            if self._finishing:
                return False
            self._finishing = True
            return True

    def start(self) -> None:
        if self.active:
            raise RuntimeError("already recording")
        self._stop.clear()
        self._cancel.clear()
        self._done.clear()
        self._finishing = False
        self._error = None
        self._wave = None
        self._peak_rms = 0.0
        self._hit_max = False
        self._cancelled = False
        self._chunks = []
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="sophon-sst-mic", daemon=True)
        self._thread.start()
        self._emit("recording")

    def stop(self) -> None:
        self._stop.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._cancel.set()
        self._stop.set()

    def join(self, timeout: float | None = 8.0) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout)
        if thread.is_alive():
            raise RuntimeError("microphone capture did not stop")

    def result(self) -> tuple[np.ndarray, int, float]:
        if self._error is not None:
            raise self._error
        if self._cancelled or self._cancel.is_set():
            raise RuntimeError("recording cancelled")
        wave = self._wave
        if wave is None or wave.size == 0:
            raise RuntimeError("no audio captured")
        duration = wave.size / float(self._rate)
        if duration < 0.12:
            raise RuntimeError("no audio captured")
        return wave, self._rate, float(self._peak_rms)

    def _emit(self, phase: str) -> None:
        cb = self._on_status
        if cb is None:
            return
        try:
            cb(phase)
        except Exception:
            pass

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            err = ImportError("sounddevice unavailable - install with: uv sync --extra sst")
            err.__cause__ = exc
            self._error = err
            self._done.set()
            return

        frames = max(1, int(round(self._rate * 0.1)))
        device = _input_device()
        kwargs: dict[str, object] = {
            "samplerate": self._rate,
            "channels": 1,
            "dtype": "float32",
            "blocksize": frames,
        }
        if device is not None:
            kwargs["device"] = device

        def callback(indata, _frames, _time_info, _status) -> None:
            if self._stop.is_set() or self._cancel.is_set():
                raise sd.CallbackStop
            mono = np.asarray(indata, dtype=np.float32).reshape(-1)
            if mono.size == 0:
                return
            rms = float(np.sqrt(np.mean(np.square(mono))))
            with self._lock:
                self._chunks.append(np.copy(mono))
                if rms > self._peak_rms:
                    self._peak_rms = rms

        try:
            with sd.InputStream(callback=callback, **kwargs):
                deadline = self._started_at + self._max_seconds
                while not self._stop.is_set() and not self._cancel.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._hit_max = True
                        self._stop.set()
                        break
                    self._stop.wait(timeout=min(0.1, remaining))
        except Exception as exc:
            callback_stop = getattr(sd, "CallbackStop", None)
            if callback_stop is None or not isinstance(exc, callback_stop):
                self._error = _capture_failure(exc)
                self._done.set()
                self._emit("stopped")
                return

        with self._lock:
            chunks = list(self._chunks)
        if chunks:
            self._wave = np.concatenate(chunks, axis=0).astype(np.float32)
        else:
            self._wave = np.zeros((0,), dtype=np.float32)
        self._done.set()
        if self._cancel.is_set():
            self._emit("cancelled")
            return
        if self._hit_max:
            self._emit("max")
            return
        self._emit("stopped")


# TODO(sst): optional Silero VAD auto-stop behind SOPHON_SST_VAD=1, off by default
# TODO(sst): live input-device picker when SOPHON_SST_MIC is unset and capture fails
