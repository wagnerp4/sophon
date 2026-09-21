from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


def _atempo_filter(speed: float) -> str:
    remaining = float(speed)
    parts: list[str] = []
    while remaining > 2.0 + 1e-6:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5 - 1e-6:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6g}")
    return ",".join(parts)


def _play_wav_native(path: Path) -> None:
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


def _play_ffplay_tempo(path: Path, speed: float) -> bool:
    ffplay = shutil.which("ffplay")
    if not ffplay:
        return False
    subprocess.run(
        [
            ffplay,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-af",
            _atempo_filter(speed),
            str(path),
        ],
        check=False,
    )
    return True


def _time_stretch_mono(wave, speed: float):
    import numpy as np

    mono = np.asarray(wave, dtype=np.float32).reshape(-1)
    if abs(float(speed) - 1.0) < 1e-3:
        return mono
    n = int(mono.shape[0])
    if n < 32:
        return mono
    win = 1024
    if n < win:
        win = max(8, (n // 2) * 2)
    hop_a = max(1, win // 4)
    hop_s = max(1, int(round(hop_a / float(speed))))
    window = np.hanning(win).astype(np.float32)
    out_len = int(np.ceil(n * hop_s / hop_a)) + win + 8
    out = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    pos_a = 0
    pos_s = 0
    while pos_a + win <= n:
        chunk = mono[pos_a : pos_a + win] * window
        out[pos_s : pos_s + win] += chunk
        wsum[pos_s : pos_s + win] += window
        pos_a += hop_a
        pos_s += hop_s
    nz = wsum > 1e-6
    out[nz] /= wsum[nz]
    end = max(pos_s, 1)
    return out[:end].astype(np.float32)


def _write_stretched_wav(path: Path, speed: float) -> Path:
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(path), always_2d=True)
    channels = [_time_stretch_mono(data[:, i], speed) for i in range(data.shape[1])]
    length = min(ch.shape[0] for ch in channels)
    stacked = np.stack([ch[:length] for ch in channels], axis=1)
    if stacked.shape[1] == 1:
        stacked = stacked[:, 0]
    fh = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    fh.close()
    dest = Path(fh.name)
    sf.write(str(dest), stacked, int(rate))
    return dest


def play_wav_file(path: Path, speed: float = 1.0) -> None:
    # TODO(speech): stop the in-flight player so overlapping /play clicks do not stack.
    rate = float(speed) if speed else 1.0
    if rate < 0.25 or rate > 8.0:
        raise ValueError("replay speed must be between 0.25 and 8")
    if abs(rate - 1.0) < 1e-3:
        _play_wav_native(path)
        return
    if _play_ffplay_tempo(path, rate):
        return
    stretched = _write_stretched_wav(path, rate)
    try:
        _play_wav_native(stretched)
    finally:
        try:
            stretched.unlink(missing_ok=True)
        except OSError:
            pass
