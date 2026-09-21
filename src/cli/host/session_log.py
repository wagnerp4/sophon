from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import TextIO


def _project_log_dir() -> Path:
    """TODO:replace and move to src/utils/logging.py"""
    from utils.device.env_bootstrap import sophon_chat_logs_dir

    return sophon_chat_logs_dir()


def make_session_log_path() -> Path:
    """TODO:replace and move to src/utils/logging.py"""
    return _project_log_dir() / "tui.log"


def resolve_session_log_path() -> Path:
    raw = os.environ.get("SOPHON_TUI_LOG", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return make_session_log_path().resolve()


class TuiSessionLog:
    """TODO:replace and move to src/utils/logging.py"""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = open(self.path, "a", encoding="utf-8")
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write("meta", f"--- session start {stamp} ---")

    def write(self, prefix: str, text: str) -> None:
        if self._file.closed:
            return
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for line in text.splitlines() or [""]:
            self._file.write(f"[{stamp}] [{prefix}] {line}\n")
        self._file.flush()

    def close(self) -> None:
        try:
            if not self._file.closed:
                stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.write("meta", f"--- session end {stamp} ---")
                self._file.close()
        except Exception:
            pass


class TeeTextIO:
    """TODO:replace and move to src/utils/logging.py"""

    def __init__(self, stream: TextIO, log: TuiSessionLog, prefix: str) -> None:
        self._stream = stream
        self._log = log
        self._prefix = prefix

    def write(self, data: str) -> int:
        if data:
            stripped = data.rstrip("\n\r")
            if stripped:
                self._log.write(self._prefix, stripped)
        return self._stream.write(data)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def open_session_log_from_env() -> TuiSessionLog | None:
    raw = os.environ.get("SOPHON_TUI_LOG", "").strip()
    if not raw:
        return None
    return TuiSessionLog(Path(raw))
