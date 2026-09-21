from __future__ import annotations

import copy

from harness.gate import Harness
from harness.policy import Policy
from harness.subagent.filter import filter_child_tools
from harness.subagent.seed import completed_turn_seed
from harness.subagent.types import DELEGATION_SCOPE, SubagentResult, SubagentRun, SubagentStartRequest


def clone_child_session(
    parent_state: object,
    *,
    depth: int,
    seed: list[dict[str, object]],
    agent_type: str,
) -> object:
    child = copy.copy(parent_state)
    child.messages = [dict(item) for item in seed]
    child.delegation_depth = int(depth)
    child.quiet = True
    child.subagent_agent_type = str(agent_type or "general")
    child.last_permission = ""
    child.shell_session = None
    child.last_subagent_run = None
    parent_harness = getattr(parent_state, "harness", None)
    if isinstance(parent_harness, Harness):
        policy = parent_harness.policy
        child.harness = Harness(
            project_root=parent_harness.project_root,
            policy=Policy(
                workspace=policy.workspace,
                mode=policy.mode,
                allow=list(policy.allow),
                ask=list(policy.ask),
                deny=list(policy.deny),
            ),
            mode_override=parent_harness.mode_override,
            session_allow=list(parent_harness.session_allow),
        )
    return child


def map_stop_reason(raw: str) -> str:
    token = str(raw or "stop").strip().lower()
    if token in ("completed", "stop", "end_turn"):
        return "completed"
    if token in ("aborted", "cancelled", "canceled"):
        return "aborted"
    if token in ("max-tokens", "max_tokens", "length"):
        return "max-tokens"
    if token in ("refusal", "content_filter"):
        return "refusal"
    if token in ("error",):
        return "error"
    return "completed"


def run_in_process(
    parent_state: object,
    request: SubagentStartRequest,
    provider: str,
    child_depth: int,
    *,
    complete_turn=None,
    session_tools=None,
    execute_tool=None,
) -> SubagentRun:
    import uuid

    from cli.agent_runtime import inject_system_extra, run_tool_loop

    if session_tools is None or execute_tool is None:
        from cli.chat import _execute_one_tool, _session_tool_bundle

        if session_tools is None:
            session_tools = _session_tool_bundle
        if execute_tool is None:
            execute_tool = _execute_one_tool

    run_id = uuid.uuid4().hex[:12]
    if provider == "fork":
        seed = completed_turn_seed(list(getattr(parent_state, "messages", None) or []))
    else:
        seed = []
    child = clone_child_session(
        parent_state,
        depth=child_depth,
        seed=seed,
        agent_type=request.agent_type,
    )
    tools, extra = session_tools(child)
    tools = filter_child_tools(list(tools or []), request.agent_type)
    call_messages = [dict(m) for m in list(getattr(child, "messages", None) or [])]
    call_messages.append({"role": "user", "content": request.prompt})
    extras = [DELEGATION_SCOPE]
    if extra:
        extras.append(extra)
    call_messages = inject_system_extra(call_messages, "\n\n".join(extras))
    try:
        outcome = run_tool_loop(
            child,
            call_messages,
            tools=tools,
            execute_tool=execute_tool,
            complete_turn=complete_turn,
        )
        text = str(outcome.text or "").strip()
        stop = map_stop_reason(outcome.stop_reason)
        diagnostic = None
    except Exception as exc:
        text = ""
        stop = "error"
        diagnostic = str(exc)
    run = SubagentRun(
        id=run_id,
        provider=provider,
        depth=child_depth,
        agent_type=request.agent_type,
        label=request.description,
        result=SubagentResult(output=text, stop_reason=stop, diagnostic=diagnostic),
        seed_len=len(seed),
    )
    setattr(parent_state, "last_subagent_run", run)
    return run
