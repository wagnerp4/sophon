from __future__ import annotations

import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

from cli.host.session_log import TuiSessionLog


def copy_text_to_system_clipboard(text: str) -> bool:
    if not text:
        return False
    commands: list[list[str]] = []
    if sys.platform == "win32":
        clip = shutil.which("clip") or shutil.which("clip.exe")
        if clip:
            commands.append([clip])
    else:
        clip = shutil.which("clip.exe")
        if clip:
            commands.append([clip])
        wsl_clip = Path("/mnt/c/Windows/System32/clip.exe")
        if wsl_clip.is_file():
            commands.append([str(wsl_clip)])
        for unix in (
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
            ["pbcopy"],
        ):
            if shutil.which(unix[0]):
                commands.append(unix)
    run_kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "check": False,
    }
    if sys.platform == "win32":
        run_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    seen: set[str] = set()
    for cmd in commands:
        key = " ".join(cmd)
        if key in seen:
            continue
        seen.add(key)
        name = Path(cmd[0]).name.lower()
        payload = text.encode("utf-16") if name in ("clip", "clip.exe") else text.encode("utf-8")
        try:
            completed = subprocess.run(cmd, input=payload, **run_kwargs)
        except OSError:
            continue
        if completed.returncode == 0:
            return True
    return False


class TuiStderrSink:
    def __init__(self, session_log: TuiSessionLog | None = None) -> None:
        self._session_log = session_log
        self.encoding = "utf-8"

    def write(self, data: str) -> int:
        if data and self._session_log is not None:
            stripped = data.rstrip("\r\n")
            if stripped:
                try:
                    self._session_log.write("err", stripped)
                except ValueError:
                    pass
        return len(data)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str):
        return getattr(sys.__stderr__, name)


@contextmanager
def capture_tui_stderr(session_log: TuiSessionLog | None = None) -> Iterator[TextIO]:
    previous = sys.stderr
    sink = TuiStderrSink(session_log)
    sys.stderr = sink  # type: ignore[assignment]
    try:
        yield previous
    finally:
        sys.stderr = previous
