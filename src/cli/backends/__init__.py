from __future__ import annotations

from typing import Literal

from cli.chat import ChatCliParams

ChatInterface = Literal["repl", "textual", "toad"]


def resolve_interface(*, tui: bool, chat_interface: str) -> ChatInterface:
    interface = str(chat_interface).lower()
    if interface == "toad":
        return "toad"
    if bool(tui) or interface == "textual":
        return "textual"
    return "repl"


def dispatch(
    interface: ChatInterface,
    params: ChatCliParams,
    *,
    spawn_window: bool,
    user_argv: list[str],
) -> None:
    if interface == "repl":
        from cli.backends import repl

        repl.run(params)
        return

    if interface == "toad":
        from cli.backends import toad

        raise SystemExit(toad.launch(user_argv))

    from cli.backends import textual
    from cli.host.spawn import is_tui_child_process, launch_tui_in_new_terminal

    if spawn_window and not is_tui_child_process():
        from utils.device.platform import desktop_terminal_available

        if desktop_terminal_available():
            raise SystemExit(launch_tui_in_new_terminal(user_argv))
        print(
            "[orodruin] No desktop terminal emulator found; running TUI in this terminal.",
            flush=True,
        )
        print("[orodruin] Tip: use --no-spawn-window to skip this message.", flush=True)

    textual.run(params)
