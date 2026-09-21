from __future__ import annotations


def completed_turn_seed(messages: list[dict[str, object]] | None) -> list[dict[str, object]]:
    out: list[dict[str, object]] = [dict(item) for item in (messages or [])]
    while out:
        last = out[-1]
        role = str(last.get("role") or "")
        if role == "tool":
            out.pop()
            continue
        if role == "assistant" and last.get("tool_calls"):
            out.pop()
            continue
        if role == "user":
            out.pop()
            continue
        break
    return out
