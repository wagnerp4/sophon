from __future__ import annotations

import json
import re
import uuid
from typing import Any

from backend.openai_compat import ToolCall

_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_FUNCTION_CALL_BLOCK = re.compile(
    r"<function_call>\s*(.*?)\s*</function_call>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_JSON = re.compile(
    r"\{[^{}]*\"name\"\s*:\s*\"([^\"]+)\"[^{}]*\"arguments\"\s*:\s*(\{.*?\}|\"[^\"]*\")[^{}]*\}",
    re.DOTALL,
)


def _args_from_raw(raw: object) -> dict[str, Any]:
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


def _call(name: str, arguments: dict[str, Any], call_id: str | None = None) -> ToolCall:
    return ToolCall(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        name=name,
        arguments=arguments,
    )


def _calls_from_mapping(item: dict[str, Any]) -> ToolCall | None:
    fn = item.get("function") if isinstance(item.get("function"), dict) else item
    name = str(fn.get("name") or item.get("name") or "").strip()
    if not name:
        return None
    args = _args_from_raw(fn.get("arguments", item.get("arguments")))
    return _call(name, args, str(item.get("id") or "") or None)


def parse_generated_tool_calls(raw_response: str, parsed: object) -> list[ToolCall]:
    if isinstance(parsed, dict):
        raw_calls = parsed.get("tool_calls")
        if isinstance(raw_calls, list):
            out: list[ToolCall] = []
            for item in raw_calls:
                if not isinstance(item, dict):
                    continue
                call = _calls_from_mapping(item)
                if call is not None:
                    out.append(call)
            if out:
                return out
        name = str(parsed.get("name") or "").strip()
        if name and ("arguments" in parsed or "parameters" in parsed):
            args = parsed.get("arguments", parsed.get("parameters"))
            return [_call(name, _args_from_raw(args))]

    text = raw_response or ""
    if isinstance(parsed, str) and parsed.strip():
        text = parsed
    out: list[ToolCall] = []
    for pattern in (_TOOL_CALL_BLOCK, _FUNCTION_CALL_BLOCK):
        for match in pattern.finditer(text):
            body = match.group(1).strip()
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                call = _calls_from_mapping(payload)
                if call is not None:
                    out.append(call)
    if out:
        return out
    for match in _TOOL_JSON.finditer(text):
        name = match.group(1).strip()
        args = _args_from_raw(match.group(2))
        if name:
            out.append(_call(name, args))
    return out


def strip_tool_call_text(text: str) -> str:
    cleaned = _TOOL_CALL_BLOCK.sub("", text or "")
    cleaned = _FUNCTION_CALL_BLOCK.sub("", cleaned)
    return cleaned.strip()
