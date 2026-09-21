from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from harness.approval import DECISION_DENY, DECISION_ONCE, DECISION_PERSIST, PermissionRequest

if TYPE_CHECKING:
    from harness.gate import Harness

_CHOICES = {
    "1": DECISION_ONCE,
    "2": DECISION_PERSIST,
    "3": DECISION_DENY,
}


def format_permission_prompt(request: PermissionRequest) -> str:
    persist = ", ".join(request.persist_rules) if request.persist_rules else request.tool
    lines = [
        f"Allow {request.tool}?",
        f"cwd: {request.cwd or '-'}",
        f"workspace: {request.workspace or '-'}",
        f"detail: {request.summary or '-'}",
        f"2 will add: {persist}",
        "1 allow this time",
        "2 add command to allow-list",
        "3 decline",
        "On Windows the command is PowerShell. Option 2 stores a prefix, not the full line.",
    ]
    if request.reason:
        lines.insert(1, f"reason: {request.reason}")
    return "\n".join(lines)


def prompt_stdio(request: PermissionRequest) -> str:
    text = format_permission_prompt(request)
    try:
        tty = bool(sys.stdin.isatty())
    except Exception:
        tty = False
    if not tty:
        return DECISION_DENY
    print(text, file=sys.stderr, flush=True)
    while True:
        try:
            raw = input("permission [1/2/3]: ").strip()
        except EOFError:
            return DECISION_DENY
        if raw in _CHOICES:
            return _CHOICES[raw]
        print("enter 1, 2, or 3", file=sys.stderr, flush=True)


def harness_system_hint(harness: "Harness") -> str:
    mode = harness.mode
    workspace = harness.workspace_path()
    if mode == "plan":
        return (
            f"Harness mode is plan. Workspace is {workspace}. "
            "You may read files and list directories inside the workspace. "
            "Do not run shell_exec or propose edits until the user runs /mode agent."
        )
    if mode == "chat":
        return (
            f"Harness mode is chat. Workspace is {workspace}. "
            "Answer from context and read tools. Do not run shell_exec or propose edits."
        )
    return (
        f"Harness policy is in effect. Workspace is {workspace}. "
        "File writes go through editor_propose_edit and wait for Review. "
        "shell_exec requires approval unless the command prefix is on the allow-list. "
        "Do not claim unrestricted disk access."
    )
