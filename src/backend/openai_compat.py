from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


def _http_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
) -> Any:
    hdrs = {"User-Agent": "orodruin-chat/0.1", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} on {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"unreachable {url}: {exc.reason}") from exc
    if not raw.strip():
        return {}
    return json.loads(raw)


def openai_compat_base_url(env_keys: tuple[str, ...], default: str) -> str:
    for key in env_keys:
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw.rstrip("/")
    return default.rstrip("/")


def list_openai_models(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout_s: float = 8.0,
) -> list[str]:
    url = f"{base_url.rstrip('/')}/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = _http_json("GET", url, headers=headers, timeout_s=timeout_s)
    items = data.get("data") if isinstance(data, dict) else None
    out: list[str] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if isinstance(mid, str) and mid.strip():
                out.append(mid.strip())
    return sorted(set(out))


def ping_openai_compat(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout_s: float = 2.0,
) -> bool:
    try:
        list_openai_models(base_url, api_key=api_key, timeout_s=timeout_s)
        return True
    except Exception:
        return False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatCompletionResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []
    out: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        args_raw = fn.get("arguments")
        args: dict[str, Any] = {}
        if isinstance(args_raw, dict):
            args = args_raw
        elif isinstance(args_raw, str) and args_raw.strip():
            try:
                parsed = json.loads(args_raw)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {"text": args_raw}
        out.append(
            ToolCall(
                id=str(item.get("id") or name),
                name=name,
                arguments=args,
            )
        )
    return out


def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _message_text(message: dict[str, Any]) -> str:
    text = _content_to_text(message.get("content")).strip()
    if text:
        return text
    for key in ("reasoning_content", "reasoning", "refusal", "text"):
        alt = _content_to_text(message.get(key)).strip()
        if alt:
            return alt
    return ""


def chat_completions(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    api_key: str | None = None,
    timeout_s: float = 600.0,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> ChatCompletionResult:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = _http_json("POST", url, body=body, headers=headers, timeout_s=timeout_s)
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("empty chat.completions response (no choices)")
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    message = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    text = _message_text(message)
    if not text:
        text = _content_to_text(choice0.get("text")).strip()
    if not text:
        text = _content_to_text(choice0.get("reasoning_content")).strip()
    tool_calls = _parse_tool_calls(message)
    finish = choice0.get("finish_reason")
    finish_reason = str(finish) if finish is not None else None
    if not text and not tool_calls:
        raise RuntimeError(
            f"empty assistant message (finish_reason={finish_reason!r}; no content/tool_calls)"
        )
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        try:
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
        except (TypeError, ValueError):
            pass
    return ChatCompletionResult(
        text=text,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
    )
