from __future__ import annotations

from dataclasses import dataclass, field

from harness.subagent.depth import (
    SubagentDepthError,
    default_max_active,
    default_max_depth,
    parent_depth_of,
    resolve_child_depth,
)
from harness.subagent.types import (
    AGENT_TYPES,
    PROVIDERS,
    SubagentResult,
    SubagentRun,
    SubagentStartRequest,
)


class SubagentActivationLimitError(Exception):
    pass


class SubagentModelOverrideError(Exception):
    pass


@dataclass
class SubagentService:
    max_depth: int = field(default_factory=default_max_depth)
    max_active: int = field(default_factory=default_max_active)
    _active: int = 0
    last_run: SubagentRun | None = None

    def start(
        self,
        provider: str,
        request: SubagentStartRequest,
        parent_state: object,
    ) -> SubagentRun:
        name = str(provider or "").strip().lower()
        if name not in PROVIDERS:
            raise ValueError(f"unknown subagent provider {provider!r}")
        agent_type = str(request.agent_type or "general").strip().lower()
        if agent_type not in AGENT_TYPES:
            raise ValueError(f"unknown agent_type {request.agent_type!r}")
        request.agent_type = agent_type
        if name == "fork" and _has_model_override(request.agent_options):
            raise SubagentModelOverrideError("fork cannot change model")
        parent_depth = request.parent_depth
        if parent_depth <= 0:
            parent_depth = parent_depth_of(parent_state)
        child_depth = resolve_child_depth(parent_depth, self.max_depth)
        if self._active >= self.max_active:
            raise SubagentActivationLimitError("ACTIVATION_LIMIT_REACHED")
        # TODO: startContinuable, send_message, interrupt_agent, list_agents, settlement notices
        # TODO: leftover-VRAM 2B classify and JEV Choice over {general, research, classify}
        self._active += 1
        try:
            from harness.subagent.driver import run_in_process

            run = run_in_process(parent_state, request, name, child_depth)
            self.last_run = run
            return run
        finally:
            self._active -= 1


def ensure_service(state: object) -> SubagentService:
    current = getattr(state, "subagent_service", None)
    if isinstance(current, SubagentService):
        return current
    service = SubagentService()
    setattr(state, "subagent_service", service)
    return service


def _has_model_override(options: dict[str, object] | None) -> bool:
    if not options:
        return False
    model = options.get("model")
    provider = options.get("provider")
    if model is not None and str(model).strip():
        return True
    if provider is not None and str(provider).strip():
        return True
    return False
