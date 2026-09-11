from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_workspace_file(path_token: str, *, roots: list[Path]) -> Path:
    raw = path_token.strip().strip("\"'")
    if not raw:
        raise ValueError("empty path")
    candidate = Path(raw).expanduser()
    tried: list[Path] = []
    if candidate.is_absolute():
        tried.append(candidate)
    else:
        for root in roots:
            tried.append((root / candidate).resolve())
    for path in tried:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "file not found. tried: " + ", ".join(str(p) for p in tried)
    )


def read_workspace_text(path_token: str, *, roots: list[Path]) -> str:
    path = resolve_workspace_file(path_token, roots=roots)
    data = path.read_text(encoding="utf-8", errors="replace")
    if not data.strip():
        raise ValueError(f"file is empty: {path}")
    return data


def message_text(msg: dict[str, object]) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def last_turn_text(
    messages: list[dict[str, object]],
    *,
    role: str = "assistant",
) -> str:
    want = role.strip().lower()
    for msg in reversed(messages):
        if str(msg.get("role") or "").lower() != want:
            continue
        text = message_text(msg).strip()
        if text:
            return text
    raise ValueError(f"no non-empty {want!r} turn in conversation")


def turn_at_index(messages: list[dict[str, object]], index: int) -> str:
    dialogue = [m for m in messages if str(m.get("role") or "") in ("user", "assistant")]
    if index < 0:
        index = len(dialogue) + index
    if index < 0 or index >= len(dialogue):
        raise IndexError(f"turn index out of range (0..{max(len(dialogue) - 1, 0)})")
    text = message_text(dialogue[index]).strip()
    if not text:
        raise ValueError(f"turn {index} is empty")
    return text


def resolve_speak_payload(
    *,
    text: str | None = None,
    path: str | None = None,
    turn: str | None = None,
    messages: list[dict[str, object]] | None = None,
    roots: list[Path] | None = None,
) -> tuple[str, str]:
    """
    Returns (spoken_text, source_label).
    Prefer path, then turn, then literal text.
    """
    if path and str(path).strip():
        body = read_workspace_text(str(path), roots=list(roots or [Path.cwd()]))
        return body, f"file:{path}"
    if turn and str(turn).strip():
        if messages is None:
            raise ValueError("turn requested but no conversation messages")
        token = str(turn).strip().lower()
        if token in ("last", "assistant", "last-assistant"):
            return last_turn_text(messages, role="assistant"), "turn:assistant"
        if token in ("user", "last-user"):
            return last_turn_text(messages, role="user"), "turn:user"
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            return turn_at_index(messages, int(token)), f"turn:{token}"
        raise ValueError("turn must be last|user|assistant|<index>")
    if text and str(text).strip():
        return str(text).strip(), "text"
    raise ValueError("provide text, path, or turn")


def tool_args_to_payload(args: dict[str, Any]) -> dict[str, str | None]:
    return {
        "text": str(args.get("text") or "").strip() or None,
        "path": str(args.get("path") or "").strip() or None,
        "turn": str(args.get("turn") or "").strip() or None,
    }
