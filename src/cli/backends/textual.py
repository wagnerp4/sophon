from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli.chat import ChatCliParams


def _require_textual() -> None:
    try:
        import textual  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Textual is not installed. Install with: pip install \"orodruin[tui]\""
        ) from exc


def run(params: "ChatCliParams") -> None:
    _require_textual()
    from cli.host.session_log import open_session_log_from_env
    from cli.tui.app import run_tui_app
    from cli.tui.io import TuiStderrSink

    session_log = open_session_log_from_env()
    if session_log is not None:
        session_log.write("meta", f"child pid={os.getpid()}")

    stderr_prev = sys.stderr
    sys.stderr = TuiStderrSink(session_log)  # type: ignore[assignment]
    try:
        run_tui_app(params, session_log)
    except KeyboardInterrupt:
        raise SystemExit(0) from None
    finally:
        sys.stderr = stderr_prev


def main() -> None:
    from cli.host.spawn import is_tui_child_process, launch_tui_in_new_terminal
    from cli.terminal import parse_chat_argv
    from utils.device.env_bootstrap import load_orodruin_dotenv

    load_orodruin_dotenv()
    argv = sys.argv[1:]
    if not is_tui_child_process() and "--no-spawn-window" not in argv:
        raise SystemExit(launch_tui_in_new_terminal(argv))

    params = parse_chat_argv(argv)
    run(params)


if __name__ == "__main__":
    main()
