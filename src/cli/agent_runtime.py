from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cli.chat_trace import ToolCallTrace, args_preview, tool_api_label


def content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def messages_for_server(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        item: dict[str, object] = {"role": role}
        if role == "tool":
            item["content"] = content_to_text(msg.get("content"))
            tcid = msg.get("tool_call_id")
            if tcid is not None:
                item["tool_call_id"] = str(tcid)
            name = msg.get("name")
            if name is not None:
                item["name"] = str(name)
        elif role == "assistant" and msg.get("tool_calls"):
            content = msg.get("content")
            if content is None:
                item["content"] = None
            else:
                item["content"] = content_to_text(content)
            item["tool_calls"] = msg["tool_calls"]
        else:
            item["content"] = content_to_text(msg.get("content"))
        out.append(item)
    return out


def inject_system_extra(
    call_messages: list[dict[str, object]], extra: str
) -> list[dict[str, object]]:
    has_system = any(m.get("role") == "system" for m in call_messages)
    if has_system:
        for m in call_messages:
            if m.get("role") == "system":
                prev = str(m.get("content") or "")
                m["content"] = (prev + "\n\n" + extra).strip()
                break
        return call_messages
    return [{"role": "system", "content": extra}, *call_messages]


def estimate_token_count(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return max(1, (len(cleaned) + 3) // 4)


def completion_token_counts(completion: object, *, text: str) -> tuple[int, int, str]:
    prompt_tokens = getattr(completion, "prompt_tokens", None)
    completion_tokens = getattr(completion, "completion_tokens", None)
    finish_reason = getattr(completion, "finish_reason", None)
    in_tok = int(prompt_tokens) if isinstance(prompt_tokens, int) else 0
    out_tok = int(completion_tokens) if isinstance(completion_tokens, int) else estimate_token_count(text)
    stop = str(finish_reason) if finish_reason else "stop"
    return in_tok, out_tok, stop


def completion_reasoning(completion: object) -> str | None:
    raw = getattr(completion, "reasoning", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def tool_calls_openai_payload(tool_calls: list) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for call in tool_calls:
        name = getattr(call, "name", None) or ""
        call_id = getattr(call, "id", None) or name
        args = getattr(call, "arguments", None)
        if not isinstance(args, dict):
            args = {}
        out.append(
            {
                "id": str(call_id),
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return out


def complete_chat_turn(state: object, messages: list[dict[str, object]], tools: list[dict] | None):
    backend_id = getattr(state, "backend_id", "hf")
    params = getattr(state, "params", None)
    max_new_tokens = int(getattr(params, "max_new_tokens", 512) or 512)
    temperature = float(getattr(params, "temperature", 0.7) or 0.7)
    top_p = float(getattr(params, "top_p", 1.0) or 1.0)
    if backend_id == "hf":
        from backend.hf.backend import hf_chat_complete

        meta = getattr(state, "meta", None)
        return hf_chat_complete(
            getattr(state, "processor", None),
            getattr(state, "model", None),
            messages,
            max_new_tokens=max_new_tokens,
            enable_thinking=bool(getattr(state, "enable_thinking", False)),
            temperature=temperature,
            top_p=top_p,
            top_k=getattr(params, "top_k", None),
            repetition_penalty=getattr(params, "repetition_penalty", None),
            seed=getattr(params, "seed", None),
            extra_specials=getattr(meta, "special_tokens", None) if meta is not None else None,
            eos_token_ids=(getattr(meta, "eos_token_ids", None) or None) if meta is not None else None,
            tools=tools,
        )
    from backend.chat_resolve import server_chat_complete

    return server_chat_complete(
        backend_id,
        model=str(getattr(state, "server_model", None) or ""),
        messages=messages_for_server(messages),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tools=tools,
    )


@dataclass
class ToolLoopOutcome:
    text: str
    elapsed_s: float
    spoke: bool
    prompt_tokens: int
    completion_tokens: int
    stop_reason: str
    reasoning: str | None
    tools: list[ToolCallTrace]
    rounds: int
    working: list[dict[str, object]] = field(default_factory=list)


def run_tool_loop(
    state: object,
    call_messages: list[dict[str, object]],
    *,
    tools: list[dict],
    execute_tool: Callable[[object, Any], str],
    complete_turn: Callable[..., Any] | None = None,
    emit_think: Callable[[str], None] | None = None,
    emit_tool: Callable[[str, dict, str, float], None] | None = None,
    emit_line: Callable[[str], None] | None = None,
) -> ToolLoopOutcome:
    complete = complete_turn or complete_chat_turn
    quiet = bool(getattr(state, "quiet", False))
    working = [dict(m) for m in call_messages]
    t0 = time.perf_counter()
    spoke = False
    text = ""
    prompt_tokens = 0
    completion_tokens = 0
    stop_reason = "stop"
    reasoning_parts: list[str] = []
    tool_rows: list[ToolCallTrace] = []
    max_rounds = getattr(state, "tool_max_rounds", None)
    round_i = 0
    while True:
        completion = complete(state, working, tools)
        text = str(getattr(completion, "text", None) or "")
        in_tok, out_tok, stop_reason = completion_token_counts(completion, text=text)
        if in_tok > 0:
            prompt_tokens = in_tok
        completion_tokens += out_tok
        thought = completion_reasoning(completion)
        if thought and thought not in reasoning_parts:
            reasoning_parts.append(thought)
            if emit_think is not None and not quiet:
                emit_think(thought)
        tool_calls = list(getattr(completion, "tool_calls", None) or [])
        if not tool_calls:
            break
        assistant_msg: dict[str, object] = {
            "role": "assistant",
            "content": text if text.strip() else None,
            "tool_calls": tool_calls_openai_payload(tool_calls),
        }
        working.append(assistant_msg)
        for call in tool_calls:
            name = str(getattr(call, "name", "") or "")
            if name == "speak":
                spoke = True
            args = getattr(call, "arguments", None)
            if not isinstance(args, dict):
                args = {}
            tool_t0 = time.perf_counter()
            result = execute_tool(state, call)
            latency = time.perf_counter() - tool_t0
            tool_rows.append(
                ToolCallTrace(
                    name=name or "unknown",
                    api=tool_api_label(name),
                    args_preview=args_preview(name, args),
                    ok=not str(result).lower().startswith("error:"),
                    latency_s=latency,
                    permission=str(getattr(state, "last_permission", "") or ""),
                )
            )
            if emit_tool is not None and not quiet:
                emit_tool(name or "unknown", args, str(result), latency)
            working.append(
                {
                    "role": "tool",
                    "tool_call_id": str(getattr(call, "id", None) or name),
                    "name": str(name),
                    "content": result,
                }
            )
        round_i += 1
        if max_rounds is not None and round_i >= max_rounds:
            if emit_line is not None and not quiet:
                emit_line("(tool loop hit max rounds; requesting final answer)")
            completion = complete(state, working, None)
            text = str(getattr(completion, "text", None) or text)
            in_tok, out_tok, stop_reason = completion_token_counts(completion, text=text)
            if in_tok > 0:
                prompt_tokens = in_tok
            completion_tokens += out_tok
            thought = completion_reasoning(completion)
            if thought and thought not in reasoning_parts:
                reasoning_parts.append(thought)
                if emit_think is not None and not quiet:
                    emit_think(thought)
            break
    elapsed = time.perf_counter() - t0
    if not str(text).strip() and spoke:
        text = "(spoken via speak tool)"
    if completion_tokens <= 0:
        completion_tokens = estimate_token_count(text)
    reasoning = "\n---\n".join(reasoning_parts) if reasoning_parts else None
    return ToolLoopOutcome(
        text=text,
        elapsed_s=elapsed,
        spoke=spoke,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stop_reason=stop_reason,
        reasoning=reasoning,
        tools=tool_rows,
        rounds=round_i,
        working=working,
    )
