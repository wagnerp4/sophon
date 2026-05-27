from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

from cli.tui_spawn import open_session_log_from_env

if TYPE_CHECKING:
    from cli.chat import ChatCliParams


def _require_textual() -> None:
    try:
        import textual  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Textual is not installed. Install with: pip install \"mithril[tui]\""
        ) from exc


def _set_console_title() -> None:
    if sys.platform == "win32" and os.environ.get("MITHRIL_TUI_CHILD"):
        try:
            os.system("title mithril")
        except Exception:
            pass


def run_chat_textual(params: "ChatCliParams") -> None:
    _set_console_title()
    _require_textual()
    session_log = open_session_log_from_env()
    if session_log is not None:
        session_log.write("meta", f"child pid={os.getpid()}")

    from cli.tui_io import TuiStderrSink

    stderr_prev = sys.stderr
    sys.stderr = TuiStderrSink(session_log)  # type: ignore[assignment]
    try:
        from cli.tui.app import run_tui_app

        run_tui_app(params, session_log)
    except KeyboardInterrupt:
        raise SystemExit(0) from None
    finally:
        sys.stderr = stderr_prev


def main() -> None:
    import sys as _sys

    from cli.tui_spawn import is_tui_child_process, launch_tui_in_new_terminal

    if not is_tui_child_process() and "--no-spawn-window" not in _sys.argv:
        raise SystemExit(launch_tui_in_new_terminal(_sys.argv[1:]))

    from cli.terminal import cli

    args = ["chat", "--tui", "--no-spawn-window", "--interface", "textual"] + [
        arg
        for arg in _sys.argv[1:]
        if arg not in ("--tui", "--no-spawn-window", "--spawn-window")
    ]
    cli.main(args=args, prog_name="mithril-chat-tui", standalone_mode=True)
