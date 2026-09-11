from __future__ import annotations

from cli.chat import ChatCliParams, run_chat


def run(params: ChatCliParams) -> None:
    run_chat(params)
