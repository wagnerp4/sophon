from __future__ import annotations

import os


class SubagentDepthError(Exception):
    pass


def default_max_depth() -> int:
    raw = os.environ.get("SOPHON_SUBAGENT_MAX_DEPTH", "1").strip()
    try:
        value = int(raw)
    except ValueError:
        return 1
    return max(0, value)


def default_max_active() -> int:
    raw = os.environ.get("SOPHON_SUBAGENT_MAX_ACTIVE", "1").strip()
    try:
        value = int(raw)
    except ValueError:
        return 1
    return max(1, value)


def parent_depth_of(state: object) -> int:
    raw = getattr(state, "delegation_depth", 0)
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def resolve_child_depth(parent_depth: int, max_depth: int) -> int:
    child_depth = int(parent_depth) + 1
    if child_depth > int(max_depth):
        raise SubagentDepthError(
            f"subagent depth {child_depth} exceeds maxDepth {max_depth}"
        )
    return child_depth
