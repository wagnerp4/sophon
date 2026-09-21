from __future__ import annotations

import os
from typing import Any

from harness.subagent.filter import SPAWN_TOOL_NAMES
from harness.subagent.types import AGENT_TYPES

SUBAGENT = "subagent"
SUBAGENT_FORK = "subagent_fork"
SUBAGENT_TOOL_NAMES = SPAWN_TOOL_NAMES

_AGENT_TYPE_ENUM = list(AGENT_TYPES)

_PROMPT_PROPS: dict[str, Any] = {
    "description": {
        "type": "string",
        "description": "Short label for this child run.",
    },
    "prompt": {
        "type": "string",
        "description": "The task for the child. Include everything it needs.",
    },
    "agent_type": {
        "type": "string",
        "enum": _AGENT_TYPE_ENUM,
        "description": "general inherits parent tools minus spawn. research is read-only.",
    },
}


def subagent_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_SUBAGENT_TOOLS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def spawn_tools_in_schema(state: object) -> bool:
    from harness.gate import PLAN_LIKE_MODES, ensure_harness

    if not subagent_tools_enabled():
        return False
    if int(getattr(state, "delegation_depth", 0) or 0) > 0:
        return False
    harness = ensure_harness(state)
    return harness.mode not in PLAN_LIKE_MODES


SUBAGENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT,
        "description": (
            "Delegate a self-contained task to a subagent with a fresh conversation. "
            "The child does not see this conversation. It returns its result, not "
            "intermediate steps. Inherit the current model."
        ),
        "parameters": {
            "type": "object",
            "properties": dict(_PROMPT_PROPS),
            "required": ["description", "prompt"],
            "additionalProperties": False,
        },
    },
}

SUBAGENT_FORK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_FORK,
        "description": (
            "Delegate a task to a subagent that inherits completed turns from this "
            "conversation (not the current in-flight turn). The child cannot change "
            "model. It returns its result, not intermediate steps."
        ),
        "parameters": {
            "type": "object",
            "properties": dict(_PROMPT_PROPS),
            "required": ["description", "prompt"],
            "additionalProperties": False,
        },
    },
}

SUBAGENT_TOOL_SYSTEM_HINT = (
    "Use subagent for independent work that should not consume this conversation. "
    "Give a complete standalone prompt. Use subagent_fork when the child must build "
    "on completed turns. Both inherit this model. Nested spawn is rejected."
)


def subagent_chat_tools(*, include: bool = True) -> list[dict[str, Any]]:
    if not include or not subagent_tools_enabled():
        return []
    return [SUBAGENT_TOOL, SUBAGENT_FORK_TOOL]


def execute_subagent_tool(parent_state: object, name: str, arguments: dict[str, Any]) -> str:
    from harness.gate import PLAN_LIKE_MODES, ensure_harness
    from harness.subagent.service import (
        SubagentActivationLimitError,
        SubagentModelOverrideError,
        ensure_service,
    )
    from harness.subagent.depth import SubagentDepthError
    from harness.subagent.types import SubagentStartRequest

    harness = ensure_harness(parent_state)
    if harness.mode in PLAN_LIKE_MODES:
        return f"error: {harness.mode} mode"
    prompt = str(arguments.get("prompt") or "").strip()
    description = str(arguments.get("description") or "").strip()
    if not prompt:
        return "error: prompt is required"
    if not description:
        return "error: description is required"
    provider = "fork" if name == SUBAGENT_FORK else "spawn"
    request = SubagentStartRequest(
        prompt=prompt,
        description=description,
        parent_depth=int(getattr(parent_state, "delegation_depth", 0) or 0),
        agent_type=str(arguments.get("agent_type") or "general"),
        agent_options=None,
    )
    service = ensure_service(parent_state)
    try:
        run = service.start(provider, request, parent_state)
    except SubagentDepthError as exc:
        return f"error: {exc}"
    except SubagentActivationLimitError:
        return "error: ACTIVATION_LIMIT_REACHED"
    except SubagentModelOverrideError as exc:
        return f"error: {exc}"
    except ValueError as exc:
        return f"error: {exc}"
    result = run.result
    if result.stop_reason != "completed":
        bits = [f"Error: {result.stop_reason}"]
        if result.diagnostic:
            bits.append(result.diagnostic)
        if result.output:
            bits.append(result.output)
        return "\n".join(bits)
    return result.output or "(empty subagent output)"
