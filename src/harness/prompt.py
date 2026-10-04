from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from harness.approval import DECISION_DENY, DECISION_ONCE, DECISION_PERSIST, PermissionRequest
from cli.key_prompt import redact_secrets

if TYPE_CHECKING:
    from harness.gate import Harness

_CHOICES = {
    "1": DECISION_ONCE,
    "2": DECISION_PERSIST,
    "3": DECISION_DENY,
}


def format_permission_prompt(
    request: PermissionRequest,
    *,
    once: str = "1",
    persist: str = "2",
    deny: str = "3",
) -> str:
    persist_rules = ", ".join(request.persist_rules) if request.persist_rules else request.tool
    lines = [
        f"Allow {request.tool}?",
        f"cwd: {request.cwd or '-'}",
        f"workspace: {request.workspace or '-'}",
        f"detail: {redact_secrets(request.summary or '-')}",
        f"{persist} will add: {redact_secrets(persist_rules)}",
        f"{once} allow this time",
        f"{persist} add command to allow-list",
        f"{deny} decline",
        "On Windows the command is PowerShell. Option 2 stores a prefix, not the full line.",
        "Left and right move. Enter confirms.",
    ]
    if request.reason:
        lines.insert(1, f"reason: {request.reason}")
    return "\n".join(lines)


def _permission_choice_map() -> dict[str, str]:
    try:
        from cli.tui.keybinds import permission_keys
    except Exception:
        return dict(_CHOICES)
    mapping = {
        "permission_once": DECISION_ONCE,
        "permission_persist": DECISION_PERSIST,
        "permission_deny": DECISION_DENY,
    }
    out: dict[str, str] = {}
    for action, decision in mapping.items():
        key = str(permission_keys().get(action, "") or "").strip().lower()
        if key:
            out[key] = decision
    for key, decision in _CHOICES.items():
        out.setdefault(key, decision)
    return out


def prompt_stdio(request: PermissionRequest) -> str:
    choices = _permission_choice_map()
    once = next((key for key, decision in choices.items() if decision == DECISION_ONCE), "1")
    persist = next((key for key, decision in choices.items() if decision == DECISION_PERSIST), "2")
    deny = next((key for key, decision in choices.items() if decision == DECISION_DENY), "3")
    text = format_permission_prompt(request, once=once, persist=persist, deny=deny)
    try:
        tty = bool(sys.stdin.isatty())
    except Exception:
        tty = False
    if not tty:
        return DECISION_DENY
    print(text, file=sys.stderr, flush=True)
    hint = "/".join(dict.fromkeys((once, persist, deny)))
    while True:
        try:
            raw = input(f"permission [{hint}]: ").strip().lower()
        except EOFError:
            return DECISION_DENY
        if raw in choices:
            return choices[raw]
        print("enter " + hint, file=sys.stderr, flush=True)


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
