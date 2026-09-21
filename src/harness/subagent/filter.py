from __future__ import annotations

from typing import Any

SPAWN_TOOL_NAMES = frozenset({"subagent", "subagent_fork"})
RESEARCH_BLOCKED_TOOLS = frozenset(
    {"shell_exec", "editor_propose_edit", "memory_propose"}
)


def tool_schema_name(entry: dict[str, Any]) -> str:
    fn = entry.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(entry.get("name") or "")


def child_blocked_names(agent_type: str) -> frozenset[str]:
    blocked = set(SPAWN_TOOL_NAMES)
    if str(agent_type or "general").strip().lower() == "research":
        blocked |= RESEARCH_BLOCKED_TOOLS
    return frozenset(blocked)


def filter_child_tools(tools: list[dict[str, Any]], agent_type: str) -> list[dict[str, Any]]:
    blocked = child_blocked_names(agent_type)
    out: list[dict[str, Any]] = []
    for entry in tools:
        name = tool_schema_name(entry)
        if name in blocked:
            continue
        out.append(entry)
    return out
