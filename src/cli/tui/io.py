from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator, TextIO

from cli.host.session_log import TuiSessionLog


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
