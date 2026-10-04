from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from typing import Any


_PLAN_ITEM = re.compile(r"^(?:\d+[\.\)]\s+|[-*]\s+)(.+)$")
_PLAN_HEAD = re.compile(r"^\s*(?:plan|planning)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class HeartbeatStep:
    kind: str
    name: str
    preview: str = ""
    path: str | None = None
    ok: bool = True
    latency_s: float = 0.0
    detail: str = ""


HEARTBEAT_KIND_ICONS: dict[str, str] = {
    "idle": "○",
    "tool": "⚙",
    "file": "▤",
    "skill": "✦",
    "permission": "⚿",
    "think": "◎",
    "context": "⊞",
}
_UNKNOWN_KIND_ICON = "◆"
_KIND_TITLE_ORDER = ("tool", "skill", "file", "permission", "context", "idle")
_KIND_WORDS: dict[str, tuple[str, str]] = {
    "tool": ("tool", "tools"),
    "file": ("file", "files"),
    "skill": ("skill", "skills"),
    "permission": ("permission", "permissions"),
    "think": ("think", "think"),
    "context": ("context", "context"),
    "idle": ("idle", "idle"),
}


def heartbeat_kind_icon(kind: str) -> str:
    key = (kind or "").strip().lower()
    if not key:
        return _UNKNOWN_KIND_ICON
    return HEARTBEAT_KIND_ICONS.get(key, _UNKNOWN_KIND_ICON)


def heartbeat_kind_count_label(kind: str, n: int) -> str:
    icon = heartbeat_kind_icon(kind)
    singular, plural = _KIND_WORDS.get(kind, (kind, kind + "s"))
    word = singular if n == 1 else plural
    if kind == "permission" and n == 1:
        return icon + " " + word
    return icon + " " + str(n) + " " + word


def heartbeat_title(steps: list[HeartbeatStep], *, running: bool, elapsed_s: float, frame: str = "") -> str:
    return heartbeat_thinking_title(steps, running=running, elapsed_s=elapsed_s, frame=frame)


def heartbeat_thinking_title(
    steps: list[HeartbeatStep],
    *,
    running: bool,
    elapsed_s: float,
    frame: str = "",
) -> str:
    counts: dict[str, int] = {}
    for step in steps:
        key = (step.kind or "").strip().lower() or "unknown"
        if key == "think":
            continue
        counts[key] = counts.get(key, 0) + 1
    bits: list[str] = ["Thinking"]
    for kind in _KIND_TITLE_ORDER:
        n = counts.pop(kind, 0)
        if n:
            bits.append(heartbeat_kind_icon(kind) + " " + str(n))
    for kind in sorted(counts):
        n = counts[kind]
        if n:
            bits.append(heartbeat_kind_icon(kind) + " " + str(n))
    summary = " · ".join(bits)
    prefix = f"{frame} " if running and frame else ""
    return f"{prefix}{summary} · {elapsed_s:.1f}s"


def format_heartbeat_step_plain(step: HeartbeatStep) -> str:
    if (step.detail or "").strip():
        return step.detail.strip()
    mark = "ok" if step.ok else "fail"
    extra = step.preview or step.path or ""
    latency = f" {step.latency_s:.2f}s" if step.latency_s > 0 else ""
    icon = heartbeat_kind_icon(step.kind)
    if extra:
        return f"{icon} [{step.kind}] {step.name} {extra}{latency} {mark}".strip()
    return f"{icon} [{step.kind}] {step.name}{latency} {mark}".strip()


@dataclass
class ToolCallTrace:
    name: str
    api: str
    args_preview: str = ""
    ok: bool = True
    latency_s: float = 0.0
    permission: str = ""


@dataclass
class RetrievalTrace:
    enabled: bool
    skipped: bool = False
    failed: bool = False
    decision: str = ""
    reason: str = ""
    backend_id: str = ""
    metrics: dict[str, Any] | None = None


@dataclass
class TurnTrace:
    timestamp: _dt.datetime
    input_tokens: int
    new_tokens: int
    gen_time_s: float
    tok_s: float
    stop_reason: str
    retrieval: RetrievalTrace | None = None
    reasoning: str | None = None
    plan_steps: list[str] = field(default_factory=list)
    tools: list[ToolCallTrace] = field(default_factory=list)
    apis: list[str] = field(default_factory=list)
    tool_rounds: int = 0


def chat_api_label(backend_id: str) -> str:
    if backend_id == "lmstudio":
        return "lmstudio POST /v1/chat/completions"
    if backend_id == "ollama":
        return "ollama POST /api/chat"
    if backend_id == "openai":
        return "openai POST /v1/chat/completions"
    if backend_id == "anthropic":
        return "anthropic POST /v1/messages"
    if backend_id == "google":
        return "google POST /v1beta/openai/chat/completions"
    if backend_id == "hf":
        return "hf generate"
    return f"{backend_id} generate"


def tool_api_label(name: str) -> str:
    mapping = {
        "vault_search": "obsidian POST /search/simple/",
        "vault_list": "obsidian GET /vault/",
        "vault_read": "obsidian GET /vault/{path}",
        "vault_recent": "obsidian GET /vault/",
        "zotero_tree": "zotero collections",
        "zotero_search": "zotero search",
        "zotero_list": "zotero collection items",
        "zotero_read": "zotero item read",
        "zotero_metrics": "zotero library metrics",
        "shell_pwd": "shell cwd",
        "shell_cd": "shell chdir",
        "shell_ls": "shell listdir",
        "shell_read": "fs read",
        "shell_exec": "shell exec",
        "editor_read": "editor buffer/fs read",
        "editor_propose_edit": "editor changeset",
        "editor_status": "editor status",
        "memory_view": "memory view",
        "memory_search": "memory search",
        "memory_read": "memory read",
        "memory_write": "memory scratch",
        "memory_propose": "memory propose fact",
        "skill_list": "skill list",
        "skill_read": "skill read",
        "skill_read_file": "skill file read",
        "speak": "tts speak",
        "transcribe": "sst transcribe",
        "subagent": "subagent spawn",
        "subagent_fork": "subagent fork",
    }
    return mapping.get(name, f"tool {name}")


def args_preview(name: str, args: dict[str, Any] | None, *, limit: int = 72) -> str:
    if not args:
        return ""
    preferred = ("query", "path", "command", "cmd", "text", "turn", "collection", "key", "description", "prompt")
    bits: list[str] = []
    for key in preferred:
        raw = args.get(key)
        if raw is None or raw == "":
            continue
        bits.append(f"{key}={_clip(str(raw), limit)}")
        if len(bits) >= 2:
            break
    if not bits:
        try:
            raw_json = json.dumps(args, ensure_ascii=False)
        except TypeError:
            raw_json = str(args)
        bits.append(_clip(raw_json, limit))
    _ = name
    return ", ".join(bits)


def plan_steps_from_text(text: str | None, *, limit: int = 8) -> list[str]:
    if not text or not text.strip():
        return []
    collecting = False
    steps: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if collecting and steps:
                break
            continue
        headed = _PLAN_HEAD.match(line)
        if headed:
            collecting = True
            rest = headed.group(1).strip()
            if rest:
                item = _PLAN_ITEM.match(rest)
                steps.append((item.group(1) if item else rest).strip())
            continue
        item = _PLAN_ITEM.match(line)
        if item:
            collecting = True
            steps.append(item.group(1).strip())
            if len(steps) >= limit:
                break
            continue
        if collecting:
            break
    return [s for s in steps if s][:limit]


def plan_steps_from_tools(tools: list[ToolCallTrace], *, limit: int = 8) -> list[str]:
    steps: list[str] = []
    for tool in tools[:limit]:
        if tool.args_preview:
            steps.append(f"{tool.name} ({tool.args_preview})")
        else:
            steps.append(tool.name)
    return steps


def unique_apis(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        label = str(item).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def build_turn_trace(
    *,
    input_tokens: int,
    new_tokens: int,
    gen_time_s: float,
    stop_reason: str,
    backend_id: str,
    retrieval: RetrievalTrace | None = None,
    reasoning: str | None = None,
    plan_text: str | None = None,
    tools: list[ToolCallTrace] | None = None,
    tool_rounds: int = 0,
    timestamp: _dt.datetime | None = None,
) -> TurnTrace:
    # TODO: persist traces with conversation history so editor replay can restore meta rows.
    # TODO: collapsible TUI widgets for long think/plan blocks (TurnHeartbeat covers live steps).
    tool_rows = list(tools or [])
    tok_s = (new_tokens / gen_time_s) if gen_time_s > 0 and new_tokens > 0 else 0.0
    think = reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None
    plan = plan_steps_from_text(plan_text)
    if not plan:
        plan = plan_steps_from_text(think)
    if not plan:
        plan = plan_steps_from_tools(tool_rows)
    apis = [chat_api_label(backend_id)]
    apis.extend(tool.api for tool in tool_rows)
    return TurnTrace(
        timestamp=timestamp or _dt.datetime.now(),
        input_tokens=max(0, int(input_tokens)),
        new_tokens=max(0, int(new_tokens)),
        gen_time_s=max(0.0, float(gen_time_s)),
        tok_s=tok_s,
        stop_reason=str(stop_reason or "stop"),
        retrieval=retrieval,
        reasoning=think,
        plan_steps=plan,
        tools=tool_rows,
        apis=unique_apis(apis),
        tool_rounds=max(0, int(tool_rounds)),
    )


def _clip(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    if limit <= 1:
        return cleaned[:limit]
    return cleaned[: limit - 1] + "…"
