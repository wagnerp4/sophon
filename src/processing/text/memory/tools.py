from __future__ import annotations

import os
from typing import Any

from .layer import MemoryLayer
from .tiers import normalize_confidence, normalize_tier

MEMORY_VIEW = "memory_view"
MEMORY_SEARCH = "memory_search"
MEMORY_READ = "memory_read"
MEMORY_WRITE = "memory_write"
MEMORY_PROPOSE = "memory_propose"

MEMORY_TOOL_NAMES = frozenset(
    {MEMORY_VIEW, MEMORY_SEARCH, MEMORY_READ, MEMORY_WRITE, MEMORY_PROPOSE}
)


def memory_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_MEMORY_TOOLS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


MEMORY_VIEW_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": MEMORY_VIEW,
        "description": (
            "List the memory directory: tier names and how many entries each holds "
            "(durable facts, procedures, session notes, earlier sessions). "
            "View memory before long tasks to see what is already known."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

MEMORY_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": MEMORY_SEARCH,
        "description": (
            "Search memory across tiers for entries matching a query. "
            "Returns entry ids, tiers, and text. Use before claiming a project fact."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search phrase."},
                "tier": {
                    "type": "string",
                    "description": "Optional tier to scope: semantic, procedural, working, episodic.",
                },
                "limit": {"type": "integer", "description": "Max hits (default 10, max 25)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

MEMORY_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": MEMORY_READ,
        "description": "Read one memory entry by tier and id.",
        "parameters": {
            "type": "object",
            "properties": {
                "tier": {
                    "type": "string",
                    "description": "semantic, procedural, working, or episodic.",
                },
                "id": {"type": "string", "description": "Entry id from memory_view/memory_search."},
            },
            "required": ["tier", "id"],
            "additionalProperties": False,
        },
    },
}

MEMORY_WRITE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": MEMORY_WRITE,
        "description": (
            "Write a session scratch note (working memory only). "
            "Durable facts require memory_propose plus the user Accepting in Review. "
            "You cannot write facts.md. "
            "Tag confidence honestly: observed (you saw evidence), decided (a choice was made), "
            "hypothesis (a guess). Never store secrets or tokens."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The note to store."},
                "confidence": {
                    "type": "string",
                    "description": "observed | decided | hypothesis (default hypothesis).",
                },
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated tags (e.g. error,repeat).",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}

MEMORY_PROPOSE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": MEMORY_PROPOSE,
        "description": (
            "Ask the user to promote scratch notes to durable facts. "
            "Queues a facts.md edit in Review. Only observed or decided notes qualify. "
            "Use after recording an error, a repeated failure, or a project decision. "
            "Then tell the user to Accept or Decline in Review."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "string",
                    "description": "Scratch entry ids, space-separated.",
                },
            },
            "required": ["ids"],
            "additionalProperties": False,
        },
    },
}

MEMORY_TOOL_SYSTEM_HINT = (
    "Memory tools persist state across sessions. Call memory_view or memory_search before "
    "repeating a failed command. Use memory_write only for scratch notes. You cannot write "
    "durable facts. After an error or the same approach twice, write an observed note "
    "(tags error,repeat), call memory_propose, and ask the user to Accept in Review. "
    "Do not retry the same failed step without a new hypothesis."
)


def memory_chat_tools() -> list[dict[str, Any]]:
    if not memory_tools_enabled():
        return []
    return [
        MEMORY_VIEW_TOOL,
        MEMORY_SEARCH_TOOL,
        MEMORY_READ_TOOL,
        MEMORY_WRITE_TOOL,
        MEMORY_PROPOSE_TOOL,
    ]


def _format_entries(entries: list) -> str:
    if not entries:
        return "(none)"
    lines = []
    for entry in entries:
        lines.append(f"[{entry.tier}] {entry.id} [{entry.confidence}] {entry.text}")
    return "\n".join(lines)


def execute_memory_tool(layer: MemoryLayer | None, name: str, arguments: dict[str, Any]) -> str:
    if layer is None:
        return "error: memory layer is disabled (SOPHON_MEMORY_DB=0)"
    if name == MEMORY_VIEW:
        status = layer.status()
        return (
            "memory tiers:\n"
            f"- semantic (durable facts): {status['semantic']}\n"
            f"- procedural: {status['procedural']}\n"
            f"- working (session notes): {status['working']}\n"
            f"- episodic (earlier sessions): {status['episodic']}"
        )
    if name == MEMORY_SEARCH:
        query = str(arguments.get("query") or "").strip()
        tier_arg = str(arguments.get("tier") or "").strip()
        tiers = [normalize_tier(tier_arg)] if tier_arg else None
        try:
            limit = int(arguments.get("limit") or 10)
        except (TypeError, ValueError):
            limit = 10
        hits = layer.search(query, tiers=tiers, limit=max(1, min(limit, 25)))
        return _format_entries(hits)
    if name == MEMORY_READ:
        tier = normalize_tier(str(arguments.get("tier") or ""))
        entry = layer.get(tier, str(arguments.get("id") or "").strip())
        if entry is None:
            return "not found"
        return f"[{entry.tier}] {entry.id} [{entry.confidence}] {entry.text}"
    if name == MEMORY_WRITE:
        text = str(arguments.get("text") or "")
        confidence = normalize_confidence(arguments.get("confidence"), "hypothesis")
        tags_raw = arguments.get("tags")
        if isinstance(tags_raw, list):
            tags = [str(t) for t in tags_raw]
        else:
            tags = [t.strip() for t in str(tags_raw or "").split(",") if t.strip()]
        _, message = layer.write_scratch(
            text, confidence=confidence, source="model", tags=tags
        )
        return message
    if name == MEMORY_PROPOSE:
        return (
            "error: memory_propose must be handled by chat (Review queue). "
            "Call this tool from the chat loop."
        )
    return f"error: unknown memory tool {name!r}"
