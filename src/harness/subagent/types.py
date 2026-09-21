from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StopReason = Literal["completed", "aborted", "error", "max-tokens", "refusal"]
AgentType = Literal["general", "research"]
ProviderName = Literal["spawn", "fork"]

AGENT_TYPES: tuple[str, ...] = ("general", "research")
PROVIDERS: tuple[str, ...] = ("spawn", "fork")
DELEGATION_SCOPE = (
    "You are a delegated subagent: your permission scope was fixed when you were started "
    "and cannot be widened from this session. Operations that require approval are rejected "
    "automatically. When the job needs access beyond that scope, do not retry the denied "
    "operation. State the limitation in your reply so the delegating agent can handle it."
)


@dataclass
class SubagentStartRequest:
    prompt: str
    description: str
    parent_depth: int = 0
    agent_type: str = "general"
    agent_options: dict[str, object] | None = None


@dataclass
class SubagentResult:
    output: str
    stop_reason: str = "completed"
    diagnostic: str | None = None


@dataclass
class SubagentRun:
    id: str
    provider: str
    depth: int
    agent_type: str
    label: str
    result: SubagentResult
    seed_len: int = 0
