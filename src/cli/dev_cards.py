from __future__ import annotations

from pathlib import Path

_BODY_CAP = 12


def line_delta(before: str, after: str) -> tuple[int, int]:
    old_lines = before.splitlines()
    new_lines = after.splitlines()
    if not before and after:
        return len(new_lines) or 1, 0
    if before and not after:
        return 0, len(old_lines) or 1
    import difflib

    added = 0
    removed = 0
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


def relative_path(path: str, root: Path | None) -> str:
    raw = (path or "").strip()
    if not raw:
        return ""
    if root is None:
        return raw
    try:
        return str(Path(raw).resolve().relative_to(Path(root).resolve()))
    except (OSError, ValueError):
        return raw


def _collapse(lines: list[str], cap: int = _BODY_CAP) -> list[str]:
    if len(lines) <= cap:
        return lines
    hidden = len(lines) - cap
    return lines[:cap] + [f"… +{hidden} lines"]


def format_diff_card(
    path: str,
    before: str,
    after: str,
    *,
    root: Path | None = None,
) -> str:
    added, removed = line_delta(before, after)
    label = relative_path(path, root) or path or "(file)"
    if not before and after:
        verb = "Added"
    elif before and not after:
        verb = "Deleted"
    else:
        verb = "Edited"
    hunk: list[str] = []
    if verb == "Added":
        hunk = [f"+{line}" for line in after.splitlines()[:_BODY_CAP]]
    elif verb == "Deleted":
        hunk = [f"-{line}" for line in before.splitlines()[:_BODY_CAP]]
    else:
        for line in before.splitlines():
            hunk.append(f"-{line}")
        for line in after.splitlines():
            hunk.append(f"+{line}")
        hunk = _collapse(hunk)
    extra = ""
    total = (len(after.splitlines()) if verb == "Added" else len(before.splitlines()))
    if verb != "Edited" and total > _BODY_CAP:
        extra = f"\n… +{total - _BODY_CAP} lines"
    body = "\n".join(hunk)
    head = f"{verb} {label} (+{added} -{removed})"
    return head if not body else head + "\n" + body + extra


def format_run_card(
    command: str,
    output: str,
    *,
    sandbox: str = "full",
    ok: bool = True,
) -> str:
    from cli.key_prompt import redact_secrets

    command = redact_secrets(command)
    output = redact_secrets(output)
    mode = sandbox if sandbox else "full"
    text = command or ""
    multiline = "\n" in text
    title = "Ran @'" if multiline else "Ran"
    head = f"{title} shell_exec · {mode}"
    if not ok:
        head += " · fail"
    cmd_lines = text.splitlines() or [text]
    if multiline:
        shown = cmd_lines[:3]
        hidden = max(0, len(cmd_lines) - len(shown))
        body = "\n".join("  │ " + line for line in shown)
        if hidden:
            body += f"\n  │ … +{hidden} lines"
    else:
        body = "  " + (cmd_lines[0] if cmd_lines else "")
    out_lines = (output or "").splitlines()
    if out_lines:
        body += "\n" + "\n".join(_collapse(["  " + line for line in out_lines]))
    return head + "\n" + body


def format_approval_line(tool: str, permission: str, detail: str = "") -> str:
    token = (permission or "").strip()
    if token == "ask:once":
        scope = "this time"
    elif token == "ask:persist":
        scope = "allow-list"
    elif token in ("ask:deny", "deny"):
        return f"Declined {tool}"
    else:
        return ""
    extra = f"({detail})" if detail else ""
    return f"You approved {tool}{extra} {scope}".replace("  ", " ")


def format_review_line(verb: str, path: str, added: int, removed: int) -> str:
    return f"{verb} {path} (+{added} -{removed})"


def format_turn_summary(text: str, paths: list[str], error_line: str) -> str:
    sentence = ""
    for chunk in (text or "").replace("\n", " ").split(". "):
        bit = chunk.strip()
        if bit:
            sentence = bit if bit.endswith(".") else bit + "."
            break
    bits = [sentence] if sentence else []
    if paths:
        bits.append("Touched " + ", ".join(paths[:4]) + ".")
    if error_line:
        bits.append(error_line.strip())
    lines = [bit for bit in bits if bit][:4]
    if not lines:
        return ""
    return "Summary: " + " ".join(lines)
