from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path


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
