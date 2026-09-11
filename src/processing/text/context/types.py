from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


@dataclass
class ContextBuildResult:
    messages: list[Message] = field(default_factory=list)
    retrieval_used: bool = False
    memory_used: bool = False
    injected_blocks: list[str] = field(default_factory=list)
