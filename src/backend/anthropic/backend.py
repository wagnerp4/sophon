from __future__ import annotations

import json
import os
from typing import Any

from backend.openai_compat import ChatCompletionResult, ToolCall, _http_json
from backend.providers import first_env, provider_spec

_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_BASE = "https://api.anthropic.com"


def anthropic_base_url() -> str:
    spec = provider_spec("anthropic")
    default = spec.default_base_url if spec is not None else _DEFAULT_BASE
    for key in ("SOPHON_ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URL"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw.rstrip("/")
    return default.rstrip("/")


def anthropic_api_key() -> str | None:
    spec = provider_spec("anthropic")
    keys = spec.env_keys if spec is not None else ("SOPHON_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    return first_env(keys)


def key_configured() -> bool:
    return bool(anthropic_api_key())


def _headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
    }


def ping_daemon(base_url: str | None = None, timeout_s: float = 2.0) -> bool:
    _ = base_url
    _ = timeout_s
    return key_configured()


def list_model_names(base_url: str | None = None, timeout_s: float = 8.0) -> list[str]:
    key = anthropic_api_key()
    if not key:
        return []
    url = f"{(base_url or anthropic_base_url()).rstrip('/')}/v1/models"
    data = _http_json("GET", url, headers=_headers(key), timeout_s=timeout_s)
    items = data.get("data") if isinstance(data, dict) else None
    out: list[str] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if isinstance(mid, str) and mid.strip():
                out.append(mid.strip())
    return out


def _openai_tools_to_anthropic(tools: list | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        schema = fn.get("parameters")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        converted.append(
            {
                "name": name,
                "description": str(fn.get("description") or ""),
                "input_schema": schema,
            }
        )
    return converted or None


def _parse_tool_arguments(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"text": raw}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _assistant_content(msg: dict[str, Any]) -> list[dict[str, Any]] | str:
    blocks: list[dict[str, Any]] = []
    text = msg.get("content")
    if isinstance(text, str) and text.strip():
        blocks.append({"type": "text", "text": text})
    raw_calls = msg.get("tool_calls")
    if isinstance(raw_calls, list):
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str((fn or {}).get("name") or call.get("name") or "").strip()
            if not name:
                continue
            args = _parse_tool_arguments((fn or {}).get("arguments") if fn else call.get("arguments"))
            cid = str(call.get("id") or name)
            blocks.append({"type": "tool_use", "id": cid, "name": name, "input": args})
    if not blocks:
        return ""
    return blocks


def openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending_tools: list[dict[str, Any]] = []

    def flush_tools() -> None:
        nonlocal pending_tools
        if pending_tools:
            out.append({"role": "user", "content": pending_tools})
            pending_tools = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        if role == "system":
            text = msg.get("content")
            if isinstance(text, str) and text.strip():
                system_parts.append(text.strip())
            continue
        if role == "tool":
            pending_tools.append(
                {
                    "type": "tool_result",
                    "tool_use_id": str(msg.get("tool_call_id") or ""),
                    "content": str(msg.get("content") or ""),
                }
            )
            continue
        flush_tools()
        if role == "assistant":
            content = _assistant_content(msg)
            out.append({"role": "assistant", "content": content})
            continue
        if role == "user":
            out.append({"role": "user", "content": str(msg.get("content") or "")})
    flush_tools()
    return "\n\n".join(system_parts).strip(), out


def _parse_anthropic_response(data: dict[str, Any]) -> ChatCompletionResult:
    blocks = data.get("content")
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("type") or "")
            if kind == "text":
                piece = block.get("text")
                if isinstance(piece, str) and piece:
                    text_parts.append(piece)
            elif kind == "tool_use":
                name = str(block.get("name") or "").strip()
                if not name:
                    continue
                args = block.get("input")
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id") or name),
                        name=name,
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
            elif kind == "thinking":
                piece = block.get("thinking")
                if isinstance(piece, str) and piece:
                    text_parts.append(piece)
    stop = data.get("stop_reason")
    finish = str(stop) if stop is not None else None
    if finish == "end_turn":
        finish = "stop"
    elif finish == "tool_use":
        finish = "tool_calls"
    elif finish == "max_tokens":
        finish = "length"
    prompt_tokens = None
    completion_tokens = None
    usage = data.get("usage")
    if isinstance(usage, dict):
        try:
            if usage.get("input_tokens") is not None:
                prompt_tokens = int(usage["input_tokens"])
            if usage.get("output_tokens") is not None:
                completion_tokens = int(usage["output_tokens"])
        except (TypeError, ValueError):
            pass
    text = "".join(text_parts).strip()
    if not text and not tool_calls:
        raise RuntimeError(
            f"empty Anthropic message (stop_reason={finish!r}; no content/tool_use)"
        )
    return ChatCompletionResult(
        text=text,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish,
        reasoning=None,
    )


def chat_complete(
    model: str,
    messages: list[dict[str, str]] | list[dict],
    *,
    max_new_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    base_url: str | None = None,
    timeout_s: float | None = None,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
):
    _ = tool_choice
    key = anthropic_api_key()
    if not key:
        from utils.device.env_bootstrap import dotenv_location_label

        raise RuntimeError(f"Set SOPHON_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY in {dotenv_location_label()}")
    deadline = 600.0
    if timeout_s is not None:
        deadline = float(timeout_s)
    else:
        raw = os.environ.get("SOPHON_ANTHROPIC_TIMEOUT_S", "").strip()
        if raw:
            deadline = float(raw)
    system_text, anth_messages = openai_messages_to_anthropic(list(messages))
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_new_tokens),
        "messages": anth_messages,
    }
    if system_text:
        body["system"] = system_text
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)
    converted = _openai_tools_to_anthropic(tools)
    if converted:
        body["tools"] = converted
    url = f"{(base_url or anthropic_base_url()).rstrip('/')}/v1/messages"
    try:
        data = _http_json(
            "POST",
            url,
            body=body,
            headers=_headers(key),
            timeout_s=deadline,
        )
    except RuntimeError as exc:
        text = str(exc).lower()
        if temperature is None or "temperature" not in text:
            raise
        body.pop("temperature", None)
        body.pop("top_p", None)
        data = _http_json(
            "POST",
            url,
            body=body,
            headers=_headers(key),
            timeout_s=deadline,
        )
    if not isinstance(data, dict):
        raise RuntimeError("Anthropic messages response was not JSON object")
    return _parse_anthropic_response(data)
