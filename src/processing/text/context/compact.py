from __future__ import annotations

import os
from typing import Any

SUMMARY_MARK = "Conversation summary (compacted):"


def keep_turns() -> int:
    raw = os.environ.get("SOPHON_COMPACT_KEEP_TURNS", "6").strip()
    try:
        value = int(raw)
    except ValueError:
        return 6
    return value if value > 0 else 6


def compact_ratio() -> float:
    raw = os.environ.get("SOPHON_COMPACT_RATIO", "0.70").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.70
    if value <= 0 or value >= 1:
        return 0.70
    return value


def env_auto_enabled() -> bool:
    raw = os.environ.get("SOPHON_COMPACT", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    chars = 0
    for message in messages:
        chars += len(str(message.get("content") or ""))
    return max(1, chars // 4)


def split_prefix_tail(
    messages: list[dict[str, Any]],
    keep: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system = [dict(item) for item in messages if item.get("role") == "system"]
    body = [dict(item) for item in messages if item.get("role") != "system"]
    user_indexes = [
        index
        for index, item in enumerate(body)
        if item.get("role") == "user"
        and not str(item.get("content") or "").startswith(SUMMARY_MARK)
    ]
    if len(user_indexes) <= keep:
        return [], system + body
    cut = user_indexes[-keep]
    return body[:cut], system + body[cut:]


def should_compact(
    messages: list[dict[str, Any]],
    *,
    context_length: int,
    keep: int | None = None,
    ratio: float | None = None,
) -> bool:
    prefix, _tail = split_prefix_tail(messages, keep if keep is not None else keep_turns())
    if not prefix:
        return False
    limit = max(int(context_length), 1) * (ratio if ratio is not None else compact_ratio())
    return estimate_tokens(messages) > limit


def summary_messages(prefix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines = ["Summarize this conversation prefix.", "List decisions, file paths, and open TODOs.", "Do not invent tool results.", ""]
    for item in prefix:
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return [
        {
            "role": "system",
            "content": (
                "You compact an older chat prefix into one note. "
                "Keep decisions, paths, and open TODOs. Do not invent tool results."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def apply_summary(summary: str, tail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    note = SUMMARY_MARK + "\n" + summary.strip()
    return [{"role": "user", "content": note}, *tail]


def estimate_summary_usd(provider: str, model: str, prefix: list[dict[str, Any]]) -> tuple[float, str]:
    from backend.energy.gate import estimate_usd

    tokens = estimate_tokens(prefix)
    return estimate_usd(provider, model, tokens, 800)
