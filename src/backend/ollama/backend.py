from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def ollama_base_url() -> str:
    raw = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    return raw.rstrip("/")


def ping_daemon(base_url: str | None = None, timeout_s: float = 5.0) -> None:
    base = base_url or ollama_base_url()
    url = f"{base}/api/version"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s):
            return
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP {exc.code} on /api/version") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama unreachable at {base}: {exc.reason}") from exc


def list_model_names(base_url: str | None = None, timeout_s: float = 10.0) -> list[str]:
    base = base_url or ollama_base_url()
    url = f"{base}/api/tags"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode())
    models = payload.get("models") or []
    out: list[str] = []
    for item in models:
        name = item.get("name")
        if isinstance(name, str) and name:
            out.append(name)
    return sorted(set(out))


def _chat_timeout_seconds(explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    raw = os.environ.get("OLLAMA_CHAT_TIMEOUT_S", "").strip()
    if raw:
        return float(raw)
    return 600.0


def chat_request_timeout_seconds() -> float:
    return _chat_timeout_seconds(None)


def _show_timeout_seconds(explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    raw = os.environ.get("OLLAMA_SHOW_TIMEOUT_S", "").strip()
    if raw:
        return float(raw)
    return 90.0


def verify_model_known(
    model: str,
    base_url: str | None = None,
    timeout_s: float | None = None,
) -> None:
    base = base_url or ollama_base_url()
    url = f"{base}/api/show"
    body = json.dumps({"model": model, "verbose": False}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deadline = _show_timeout_seconds(timeout_s)
    try:
        with urllib.request.urlopen(req, timeout=deadline) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"Ollama does not expose model {model!r} (daemon at {base}). HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RuntimeError(
                f"Ollama /api/show timed out after {deadline}s for {model!r}. "
                "Increase OLLAMA_SHOW_TIMEOUT_S if cloud registry resolves slowly."
            ) from exc
        raise RuntimeError(f"Ollama unreachable at {base}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama /api/show timed out after {deadline}s for {model!r}. "
            "Increase OLLAMA_SHOW_TIMEOUT_S if cloud registry resolves slowly."
        ) from exc


def _text_from_ollama_message(msg: object) -> str:
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        if content.strip():
            return content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        joined = "".join(parts).strip()
        if joined:
            return joined
    thinking = msg.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        return thinking.strip()
    return "" if isinstance(content, str) else ""


def chat_complete(
    model: str,
    messages: list[dict[str, object]],
    *,
    max_new_tokens: int,
    base_url: str | None = None,
    timeout_s: float | None = None,
) -> str:
    base = base_url or ollama_base_url()
    url = f"{base}/api/chat"
    body_obj: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_new_tokens},
    }
    body = json.dumps(body_obj).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deadline = _chat_timeout_seconds(timeout_s)
    try:
        with urllib.request.urlopen(req, timeout=deadline) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RuntimeError(
                f"Ollama chat timed out after {deadline}s. "
                "Increase OLLAMA_CHAT_TIMEOUT_S if cloud models need longer."
            ) from exc
        raise RuntimeError(f"Ollama unreachable at {base}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama chat timed out after {deadline}s. "
            "Increase OLLAMA_CHAT_TIMEOUT_S if cloud models need longer."
        ) from exc
    msg = data.get("message") or {}
    text = _text_from_ollama_message(msg)
    if text == "" and isinstance(msg, dict):
        if msg.get("tool_calls") or msg.get("role"):
            raise RuntimeError(f"Ollama returned no assistant text: {data!r}")
        raise RuntimeError(f"Unexpected Ollama response shape: {data!r}")
    return text
