from __future__ import annotations

import datetime as _dt
import math
import re
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.markup import escape as _escape_markup

from backend.chat_resolve import is_server_backend
from backend.hf.registry import HF_MODEL_PRESETS
from cli.chat_trace import HeartbeatStep, RetrievalTrace, TurnTrace, format_heartbeat_step_plain

if TYPE_CHECKING:
    from cli.chat import _SessionState

_REASONING_MAX_CHARS = 720
_REASONING_MAX_LINES = 8
_META_LABEL_WIDTH = 10


_BLUE = "#60a5fa"
_BLUE_DIM = "#93c5fd"
_BLUE_DEEP = "#3b82f6"
_NEXUS_BLUE = (
    "#bae6fd",
    "#7dd3fc",
    "#38bdf8",
    "#1d4ed8",
    "#172554",
)

_MODE_COLORS = {
    "chat": "#22c55e",
    "agent": "#f8fafc",
    "plan": "#eab308",
}

_NEXUS_WORDMARK = (
    "███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗",
    "████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝",
    "██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗",
    "██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║",
    "██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║",
)
_NEXUS_TAGLINE = "harness · control central"
SESSION_SEPARATOR = "─" * 32


def nexus_wordmark_markup() -> str:
    from utils.device.env_bootstrap import sophon_agent_root, sophon_version_string

    parts: list[str] = []
    for i, body in enumerate(_NEXUS_WORDMARK):
        color = _NEXUS_BLUE[min(i, len(_NEXUS_BLUE) - 1)]
        parts.append(f"[bold {color}]{_escape_markup(body)}[/]")
    version = sophon_version_string().lstrip("v")
    tag = f"{_NEXUS_TAGLINE}  v{version}"
    parts.append(f"[dim]{_escape_markup(tag)}[/]")
    root = str(sophon_agent_root())
    parts.append(f"[dim]{_escape_markup(root)}[/]")
    parts.append(f"[dim]{_escape_markup(SESSION_SEPARATOR)}[/]")
    return "\n".join(parts)

ESSENTIAL_SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "full command list (also ? or F1 in TUI)"),
    ("/quit", "exit chat (q or Ctrl+Q in TUI)"),
    ("/speak TEXT", "speak now (Pipecat Kokoro default)"),
    ("/transcribe PATH", "transcribe audio with Qwen ASR"),
    ("/listen", "toggle mic (Speak / Ctrl+L). /listen stop or /listen-cancel"),
    ("/play [clip] [speed]", "replay a stored speech clip (1x 1.5x 2x)"),
    ("/clips", "list stored speech clips for this session"),
    ("/tts on|off", "auto-speak assistant replies"),
    ("/backend", "auto | ollama | lmstudio | hf | openai | anthropic | google"),
    ("/models", "open model picker (TUI) or list sources"),
    ("/models-sync", "refresh versioned OpenAI/Anthropic/Google model index"),
    ("/model PRESET", "switch model (or server model id)"),
    ("/params", "show sampling / max tokens (change with /temp /max …)"),
    ("/tool-rounds", "tool-loop rounds (N or unlimited); /unlimited to uncap"),
    ("/model-download PRESET", "fetch weights from Hugging Face Hub"),
    ("/setup", "device recommender: top 10 then install rank 1 or skip"),
    ("/reset", "clear conversation, keep system prompt"),
    ("/save [path]", "export transcript"),
    ("/tokens", "prompt tokens vs context window"),
    ("/regen", "redo last reply (r)"),
    ("/rag-status", "retrieval backend status"),
    ("/tools", "show enabled speak / vault / zotero / shell tools"),
    ("/log", "session log path and tail. /log path | /log [N]"),
    ("/mode", "harness mode: plan | chat | agent"),
    ("/permissions", "show harness allow/ask/deny rules"),
    ("/chat", "leave ! shell mode"),
    ("/auto-attach", "on|off attach @path file text (default on)"),
)


@dataclass(frozen=True)
class TranscriptLine:
    plain: str
    markup: str


def resolve_banner_width(width: int | None = None) -> int:
    if width is not None and width >= 40:
        return min(int(width), 240)
    try:
        cols = int(shutil.get_terminal_size(fallback=(100, 24)).columns)
    except Exception:
        cols = 100
    return min(max(cols - 4, 40), 240)


def _fit(text: str, width: int) -> str:
    text = text.rstrip()
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _box_outer(width: int, *, top: bool) -> str:
    if width < 2:
        return "═" * max(width, 0)
    if top:
        return "╔" + ("═" * (width - 2)) + "╗"
    return "╚" + ("═" * (width - 2)) + "╝"


def _box_rule(width: int) -> str:
    if width < 2:
        return "─" * max(width, 0)
    return "╠" + ("═" * (width - 2)) + "╣"


def _box_row(text: str, width: int) -> str:
    if width < 2:
        return _fit(text, width)
    inner = width - 2
    return "║" + _fit(text, inner).ljust(inner)[:inner] + "║"


def _box_center(text: str, width: int) -> str:
    if width < 2:
        return _fit(text, width)
    inner = width - 2
    body = _fit(text.strip(), inner).center(inner)
    return "║" + body + "║"


def _center_block_rows(rows: tuple[str, ...] | list[str], inner: int) -> list[str]:
    cleaned = [row.rstrip() for row in rows]
    block_w = max((len(row) for row in cleaned), default=0)
    block_w = min(block_w, inner) if inner > 0 else 0
    pad = max((inner - block_w) // 2, 0)
    out: list[str] = []
    for row in cleaned:
        body = row[:block_w].ljust(block_w)
        line = (" " * pad) + body
        out.append(line[:inner].ljust(inner))
    return out


def _join_bit_rows(bits: list[str], width: int) -> list[str]:
    rows: list[str] = []
    current = ""
    for bit in bits:
        bit = bit.strip()
        if not bit:
            continue
        piece = bit if not current else " · " + bit
        if current and len(current) + len(piece) > width:
            rows.append(current)
            current = bit
            continue
        current = current + piece if current else bit
    if current:
        rows.append(current)
    return rows


def _framed_inner_markup(framed: str, inner_markup: str, *, frame_color: str) -> str:
    if len(framed) < 2:
        return inner_markup
    return (
        f"[bold {frame_color}]{_escape_markup(framed[0])}[/]"
        + inner_markup
        + f"[bold {frame_color}]{_escape_markup(framed[-1])}[/]"
    )


def _model_identity(state: _SessionState) -> str:
    from cli.chat import chat_session_has_weights, chat_session_ready

    if is_server_backend(state.backend_id):
        model = (state.server_model or "").strip()
        if not model:
            return f"{state.backend_id} · no model selected"
        if chat_session_ready(state):
            return f"{state.backend_id} · {model}"
        return f"{state.backend_id} · {model} · not loaded"
    if state.preset_key:
        label = state.preset_key
    elif state.model_path.strip():
        label = state.model_path.replace("\\", "/").rstrip("/").split("/")[-1] or state.model_path
    else:
        label = "(no model)"
    if chat_session_has_weights(state):
        return f"hf · {label}"
    return f"hf · {label} · not loaded (startup default)"


def model_identity_line(state: _SessionState) -> str:
    return _model_identity(state)


def _gen_params_summary(state: _SessionState) -> str:
    p = state.params
    parts: list[str] = [f"max={p.max_new_tokens}"]
    if p.temperature is not None:
        parts.append(f"temp={p.temperature}")
    if p.top_p is not None:
        parts.append(f"top_p={p.top_p}")
    if p.top_k is not None:
        parts.append(f"top_k={p.top_k}")
    if p.repetition_penalty is not None and p.repetition_penalty != 1.0:
        parts.append(f"rep={p.repetition_penalty}")
    if p.seed is not None:
        parts.append(f"seed={p.seed}")
    return "  ".join(parts)


def _context_summary(state: _SessionState) -> str:
    if is_server_backend(state.backend_id):
        return "ctx: server"
    meta = state.meta
    if meta is not None and meta.max_position_embeddings:
        return f"ctx: {meta.max_position_embeddings:,}"
    return ""


def _model_summary_lines(state: _SessionState) -> list[str]:
    lines: list[str] = [f"backend: {state.backend_id}"]
    if is_server_backend(state.backend_id):
        lines.append(f"model: {state.server_model or '(none)'}")
        ctx = _context_summary(state)
        if ctx:
            lines.append(ctx)
        params = _gen_params_summary(state)
        if params:
            lines.append(params)
        return lines
    if state.preset_key:
        preset = HF_MODEL_PRESETS[state.preset_key]
        lines.append(f"model: {state.preset_key} - {preset.repo_id}")
        lines.append(f"path: {state.model_path}")
    elif state.model_path.strip():
        lines.append(f"model path: {state.model_path}")
    ctx = _context_summary(state)
    if ctx:
        lines.append(ctx)
    params = _gen_params_summary(state)
    if params:
        lines.append(params)
    if state.quantization and state.quantization != "none":
        lines.append(f"quantization: {state.quantization}")
    return lines


def _status_integration_bits(state: _SessionState) -> list[str]:
    from integrations.obsidian.client import obsidian_tools_enabled
    from integrations.zotero.client import zotero_tools_enabled

    bits: list[str] = []
    if state.retriever is not None:
        bits.append(f"rag {state.retriever.backend_id()}")
    bits.append(f"sst {'on' if state.sst_enabled else 'off'}")
    bits.append(f"obsidian {'on' if obsidian_tools_enabled() else 'off'}")
    bits.append(f"zotero {'on' if zotero_tools_enabled() else 'off'}")
    try:
        from integrations.google.oauth import google_tools_enabled, list_accounts
        from integrations.google.bookmarks import bookmarks_available
        from integrations.google.search import web_search_tools_enabled

        google_on = (
            (google_tools_enabled() and bool(list_accounts()))
            or bookmarks_available()
            or web_search_tools_enabled()
        )
        bits.append(f"google {'on' if google_on else 'off'}")
    except Exception:
        bits.append("google off")
    bits.append(f"tts {'on' if state.tts_enabled else 'off'}")
    return bits


def _status_runtime_bits(state: _SessionState) -> list[str]:
    from integrations.shell.runner import shell_tools_enabled

    bits = [f"shell {'on' if shell_tools_enabled() else 'off'}"]
    if state.memory is not None and state.memory_scope is not None:
        bits.append(f"mem {state.memory_scope.session_id}")
    stats = _runtime_stats_summary(state)
    if stats:
        bits.append(stats)
    return bits


def _status_bits(state: _SessionState) -> str:
    return " · ".join(_status_integration_bits(state) + _status_runtime_bits(state))


def _tools_status_bits() -> str:
    from integrations.obsidian.client import obsidian_tools_enabled
    from integrations.shell.runner import shell_tools_enabled
    from integrations.zotero.client import zotero_tools_enabled

    google_label = "google off"
    try:
        from integrations.google.oauth import google_tools_enabled, list_accounts
        from integrations.google.bookmarks import bookmarks_available
        from integrations.google.search import web_search_tools_enabled

        google_on = (
            (google_tools_enabled() and bool(list_accounts()))
            or bookmarks_available()
            or web_search_tools_enabled()
        )
        google_label = f"google {'on' if google_on else 'off'}"
    except Exception:
        pass
    return " · ".join(
        [
            f"obsidian {'on' if obsidian_tools_enabled() else 'off'}",
            f"zotero {'on' if zotero_tools_enabled() else 'off'}",
            google_label,
            f"shell {'on' if shell_tools_enabled() else 'off'}",
        ]
    )


def _inventory_line(state: _SessionState) -> str:
    from cli.chat import harness_inventory_counts

    n_skills, n_tools, n_mcp, n_injected = harness_inventory_counts(state)
    return (
        f" Skills: {n_skills} · NexusTools: {n_tools} · MCP: {n_mcp} · Injected: {n_injected}"
    )


def _harness_mode_name(state: _SessionState) -> str:
    from harness import ensure_harness

    try:
        mode = str(ensure_harness(state).mode or "").strip().lower()
    except Exception:
        return "agent"
    if mode in _MODE_COLORS:
        return mode
    return "agent"


def _mode_banner_plain(mode: str) -> str:
    return f" mode {mode}"


def _runtime_stats_summary(state: _SessionState) -> str:
    return state.stats.format_banner_stats()


_HEADER_CMDS = (
    "/help · /quit · /tools · /speak · /listen · /transcribe · "
    "/tts · /sst · /backend · /models · /params · F1"
)


def chat_session_header_lines(
    state: _SessionState,
    *,
    for_markup: bool = False,
    width: int | None = None,
) -> list[TranscriptLine]:
    from cli.chat import chat_session_ready

    w = resolve_banner_width(width)
    lines: list[TranscriptLine] = []

    def add(plain: str, markup: str | None = None) -> None:
        text = _fit(plain.strip(), w)
        if not text:
            return
        if for_markup and markup:
            lines.append(TranscriptLine(plain=text, markup=markup))
            return
        if for_markup:
            lines.append(TranscriptLine(plain=text, markup=f"[dim]{_escape_markup(text)}[/]"))
            return
        lines.append(TranscriptLine(plain=text, markup=text))

    if not chat_session_ready(state):
        if is_server_backend(state.backend_id):
            warn = f"no {state.backend_id} model selected · /models"
        else:
            warn = "no local weights loaded · startup default · /model-download or /models"
        add(warn, f"[yellow]{_escape_markup(warn)}[/]" if for_markup else None)

    identity_bits = [_model_identity(state)]
    identity_bits.extend(p for p in (_gen_params_summary(state), _context_summary(state)) if p)
    identity = " · ".join(identity_bits)
    add(identity, f"[bold]{_escape_markup(identity)}[/]" if for_markup else None)
    inventory = _inventory_line(state).strip()
    add(inventory, f"[{_BLUE_DIM}]{_escape_markup(inventory)}[/]" if for_markup else None)
    add(_HEADER_CMDS, f"[{_BLUE_DIM}]{_escape_markup(_HEADER_CMDS)}[/]" if for_markup else None)
    integ = " · ".join(_status_integration_bits(state))
    if integ:
        add(integ)
    runtime = " · ".join(_status_runtime_bits(state))
    if runtime:
        add(runtime)
    mode = _harness_mode_name(state)
    from backend.energy.ledger import sum_today
    from backend.energy.regime import ensure_energy, status_line

    energy = status_line(ensure_energy(state), spent_today=sum_today(ensure_energy(state)))
    mode_plain = f"mode {mode} · {energy}"
    mode_color = _MODE_COLORS.get(mode, _BLUE_DIM)
    mode_markup = (
        f"[dim]mode [/][bold {mode_color}]{_escape_markup(mode)}[/] [dim]{_escape_markup(energy)}[/]"
        if for_markup
        else None
    )
    add(mode_plain, mode_markup)
    return lines


def chat_banner_lines(
    state: _SessionState,
    *,
    log_path: str | None = None,
    started_at: _dt.datetime | None = None,
    for_markup: bool = False,
    width: int | None = None,
) -> list[str]:
    _ = log_path
    _ = started_at
    return [
        line.plain
        for line in chat_session_header_lines(
            state,
            for_markup=for_markup,
            width=width,
        )
    ]


def chat_banner_transcript_lines(
    state: _SessionState,
    *,
    log_path: str | None = None,
    started_at: _dt.datetime | None = None,
    for_markup: bool = False,
    width: int | None = None,
) -> list[TranscriptLine]:
    _ = log_path
    _ = started_at
    return chat_session_header_lines(state, for_markup=for_markup, width=width)


def format_user_turn_line(
    text: str,
    *,
    turn: int,
    timestamp: _dt.datetime | None = None,
    for_markup: bool = False,
) -> TranscriptLine:
    ts = (timestamp or _dt.datetime.now()).strftime("%H:%M:%S")
    plain = f"[{ts}] turn {turn:>3}  you> {text}"
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    markup = (
        f"[dim]{ts}[/] [dim]turn {turn:>3}[/]  "
        f"[bold {_BLUE}]you>[/] {_escape_markup(text)}"
    )
    return TranscriptLine(plain=plain, markup=markup)


def format_assistant_turn_header(
    *,
    turn: int,
    timestamp: _dt.datetime | None = None,
    for_markup: bool = False,
    token_summary: str | None = None,
) -> TranscriptLine:
    ts = (timestamp or _dt.datetime.now()).strftime("%H:%M:%S")
    extra = f"  {token_summary}" if token_summary else ""
    plain = f"[{ts}] turn {turn:>3}  nexus>{extra}"
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    extra_markup = f" [dim]{_escape_markup(token_summary)}[/]" if token_summary else ""
    markup = (
        f"[dim]{ts}[/] [dim]turn {turn:>3}[/]  "
        f"[bold {_BLUE_DIM}]nexus>[/]{extra_markup}"
    )
    return TranscriptLine(plain=plain, markup=markup)


def compact_token_summary(trace: TurnTrace) -> str:
    parts: list[str] = []
    if trace.input_tokens > 0:
        parts.append(f"in={trace.input_tokens}")
    if trace.new_tokens > 0:
        parts.append(f"out={trace.new_tokens}")
    if trace.tok_s > 0:
        parts.append(f"{trace.tok_s:.1f} tok/s")
    if trace.gen_time_s > 0:
        parts.append(f"{trace.gen_time_s:.2f}s")
    if trace.stop_reason:
        parts.append(f"stop={trace.stop_reason}")
    return " · ".join(parts)


def format_assistant_turn_lines(
    text: str,
    *,
    turn: int,
    timestamp: _dt.datetime | None = None,
    trace: TurnTrace | None = None,
    for_markup: bool = False,
    include_trace_details: bool = True,
) -> list[TranscriptLine]:
    stamp = timestamp
    if stamp is None and trace is not None:
        stamp = trace.timestamp
    summary = compact_token_summary(trace) if trace is not None else None
    lines = [
        format_assistant_turn_header(
            turn=turn,
            timestamp=stamp,
            for_markup=for_markup,
            token_summary=summary,
        )
    ]
    if trace is not None and include_trace_details:
        lines.extend(
            format_turn_trace_lines(
                trace,
                for_markup=for_markup,
                include_tokens=False,
                reply_text=text,
            )
        )
    original_lines = (text or "").splitlines() or [""]
    if for_markup:
        display_lines = fluent_assistant_display(text).splitlines() or [""]
        n = max(len(original_lines), len(display_lines))
        while len(original_lines) < n:
            original_lines.append("")
        while len(display_lines) < n:
            display_lines.append("")
        for orig, display in zip(original_lines, display_lines):
            rendered = format_assistant_body_line(display, for_markup=True)
            lines.append(TranscriptLine(plain=orig, markup=rendered.markup))
    else:
        for orig in original_lines:
            lines.append(format_assistant_body_line(orig, for_markup=False))
    if trace is not None:
        lines.append(format_duration_footer(trace, for_markup=for_markup, timestamp=stamp))
    return lines


def format_turn_trace_lines(
    trace: TurnTrace,
    *,
    for_markup: bool = False,
    include_tokens: bool = True,
    reply_text: str | None = None,
) -> list[TranscriptLine]:
    rows: list[tuple[str, str]] = []
    if include_tokens:
        token_bits: list[str] = []
        if trace.input_tokens > 0:
            token_bits.append(f"in={trace.input_tokens}")
        else:
            token_bits.append("in=?")
        token_bits.append(f"out={trace.new_tokens}")
        if trace.tok_s > 0:
            token_bits.append(f"{trace.tok_s:.1f} tok/s")
        token_bits.append(f"{trace.gen_time_s:.2f}s")
        token_bits.append(f"stop={trace.stop_reason}")
        rows.append(("tokens", " · ".join(token_bits)))
    retrieval_line = _format_retrieval_trace(trace.retrieval)
    if retrieval_line:
        rows.append(("retrieval", retrieval_line))
    if trace.tools:
        tool_bits: list[str] = []
        for tool in trace.tools:
            label = tool.name
            if tool.args_preview:
                label = f"{tool.name} ({tool.args_preview})"
            if tool.permission:
                label += f" {tool.permission}"
            if not tool.ok:
                label += " fail"
            if tool.latency_s > 0:
                label += f" {tool.latency_s:.2f}s"
            tool_bits.append(label)
        tool_line = " · ".join(tool_bits)
        if trace.tool_rounds > 0:
            tool_line += f" · {trace.tool_rounds} round" + ("s" if trace.tool_rounds != 1 else "")
        rows.append(("tools", tool_line))
    if trace.apis:
        rows.append(("apis", " · ".join(trace.apis)))
    if trace.plan_steps:
        numbered = "  ".join(f"{i}. {step}" for i, step in enumerate(trace.plan_steps, 1))
        rows.append(("plan", numbered))
    lines: list[TranscriptLine] = [_format_meta_row(label, body, for_markup=for_markup) for label, body in rows]
    reasoning = trace.reasoning.strip() if isinstance(trace.reasoning, str) else ""
    if reasoning and (reply_text is None or reasoning != reply_text.strip()):
        think_lines = _truncate_reasoning(reasoning)
        if think_lines:
            lines.append(_format_meta_row("think", think_lines[0], for_markup=for_markup))
            for extra in think_lines[1:]:
                lines.append(_format_meta_row("", extra, for_markup=for_markup))
    return lines


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_STAR_LIST_RE = re.compile(r"^(\s*)(?:[*•])(\s+)")
_BULLET_PREFIX_RE = re.compile(r"^(\s*[-*]\s+)(.*)$")
_MD_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_MD_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)")
_MD_ITALIC_UNDER_RE = re.compile(r"(?<![A-Za-z0-9_])_([^_]+)_(?![A-Za-z0-9_])")


def fluent_assistant_display(text: str) -> str:
    lines: list[str] = []
    for line in (text or "").splitlines() or [""]:
        cleaned = _HEADING_RE.sub("", line)
        cleaned = _STAR_LIST_RE.sub(r"\1- ", cleaned)
        lines.append(cleaned)
    return "\n".join(lines)


def chat_prompt_placeholder(state: object | None) -> str:
    if state is not None and bool(getattr(state, "shell_mode", False)):
        return "Shell command (! or /chat to leave)"
    return "Message, /command, @path, or ! for shell"


def format_duration_footer(
    trace: TurnTrace,
    *,
    for_markup: bool = False,
    timestamp: _dt.datetime | None = None,
) -> TranscriptLine:
    stamp = timestamp or trace.timestamp or _dt.datetime.now()
    seconds = math.ceil(max(float(trace.gen_time_s), 0.0))
    clock = stamp.strftime("%H:%M")
    plain = f"Duration: {seconds}s. Time: {clock}"
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    return TranscriptLine(plain=plain, markup=f"[dim]{plain}[/]")


def markdown_inline_to_rich(text: str) -> str:
    from cli.path_highlights import chip_markup, color_for_span

    prefix = ""
    body = text
    bullet = _BULLET_PREFIX_RE.match(text)
    if bullet:
        prefix = _escape_markup(bullet.group(1))
        body = bullet.group(2)
    held: list[str] = []

    def _hold(markup: str) -> str:
        held.append(markup)
        return "\x00" + str(len(held) - 1) + "\x00"

    out = _escape_markup(body)

    def _code_sub(match: re.Match[str]) -> str:
        inner = match.group(1)
        color = color_for_span(inner)
        return _hold(chip_markup(inner, color))

    out = _MD_CODE_RE.sub(_code_sub, out)
    out = _MD_BOLD_RE.sub(lambda m: _hold("[bold]" + m.group(2) + "[/bold]"), out)
    out = _MD_ITALIC_STAR_RE.sub(lambda m: _hold("[italic]" + m.group(1) + "[/italic]"), out)
    out = _MD_ITALIC_UNDER_RE.sub(lambda m: _hold("[italic]" + m.group(1) + "[/italic]"), out)
    for i in range(len(held) - 1, -1, -1):
        out = out.replace("\x00" + str(i) + "\x00", held[i])
    return prefix + out


def format_assistant_body_line(text: str, *, for_markup: bool = False) -> TranscriptLine:
    plain = text
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    return TranscriptLine(plain=plain, markup="  " + markdown_inline_to_rich(text))


def format_heartbeat_step_line(step: HeartbeatStep, *, for_markup: bool = False) -> TranscriptLine:
    plain = format_heartbeat_step_plain(step)
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    return TranscriptLine(plain=plain, markup=f"[dim]{_escape_markup(plain)}[/]")


def format_system_line(text: str, *, for_markup: bool = False) -> TranscriptLine:
    plain = text
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    if text.startswith("(") and text.endswith(")"):
        return TranscriptLine(plain=plain, markup=f"[dim italic]{_escape_markup(text)}[/]")
    return TranscriptLine(plain=plain, markup=f"[dim]{_escape_markup(text)}[/]")


def format_speech_clip_line(clip: object, *, for_markup: bool = False) -> TranscriptLine:
    from processing.audio.speech.session_store import SpeechClip, default_replay_speeds, format_replay_speed

    if not isinstance(clip, SpeechClip):
        plain = str(clip)
        return TranscriptLine(plain=plain, markup=plain)
    who = "you" if clip.role == "user" else "nexus"
    speeds = "  ".join(f"/play {clip.clip_id} {format_replay_speed(s)}" for s in default_replay_speeds())
    plain = f"  [clip {clip.clip_id} {who} {clip.duration_s:.1f}s]  {speeds}"
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    markup = f"[dim]{_escape_markup(plain)}[/]"
    return TranscriptLine(plain=plain, markup=markup)


def _format_meta_row(label: str, body: str, *, for_markup: bool) -> TranscriptLine:
    if label:
        pad = " " * max(_META_LABEL_WIDTH - len(label), 1)
        plain = f"  {label}{pad}{body}"
    else:
        pad = " " * (_META_LABEL_WIDTH + 2)
        plain = f"{pad}{body}"
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    if label:
        markup = (
            f"[dim]  {_escape_markup(label)}{pad}[/]"
            f"[dim]{_escape_markup(body)}[/]"
        )
    else:
        markup = f"[dim]{_escape_markup(plain)}[/]"
    return TranscriptLine(plain=plain, markup=markup)


def _format_retrieval_trace(retrieval: RetrievalTrace | None) -> str:
    if retrieval is None:
        return ""
    if not retrieval.enabled:
        return "off"
    if retrieval.failed:
        reason = retrieval.reason or "error"
        return f"failed · {reason}"
    if retrieval.skipped:
        reason = retrieval.reason or retrieval.decision or "skip"
        return f"skipped · {reason}"
    bits: list[str] = []
    metrics = retrieval.metrics if isinstance(retrieval.metrics, dict) else None
    if metrics:
        try:
            from processing.text.retrieval.metrics import RetrievalQueryMetrics, format_query_metrics

            bits.append(
                format_query_metrics(
                    RetrievalQueryMetrics(
                        latency_s=float(metrics.get("latency_s") or 0.0),
                        n_hits=int(metrics.get("n_hits") or 0),
                        top_k=int(metrics.get("top_k") or 0),
                        query_chars=int(metrics.get("query_chars") or 0),
                        top_score=metrics.get("top_score"),
                        mean_score=metrics.get("mean_score"),
                        min_score=metrics.get("min_score"),
                        score_margin=metrics.get("score_margin"),
                        backend_id=str(metrics.get("backend_id") or retrieval.backend_id or ""),
                    )
                )
            )
        except Exception:
            backend = str(metrics.get("backend_id") or retrieval.backend_id or "")
            hits = metrics.get("n_hits")
            top_k = metrics.get("top_k")
            latency = metrics.get("latency_s")
            chunk = []
            if backend:
                chunk.append(backend)
            if hits is not None and top_k is not None:
                chunk.append(f"hits={hits}/{top_k}")
            if latency is not None:
                chunk.append(f"latency={float(latency):.3f}s")
            bits.append(" ".join(chunk) if chunk else str(metrics))
    elif retrieval.backend_id:
        bits.append(retrieval.backend_id)
    if retrieval.decision:
        bits.append(retrieval.decision)
    if retrieval.reason and retrieval.decision != retrieval.reason:
        bits.append(retrieval.reason)
    return " · ".join(b for b in bits if b) or "ran"


def _truncate_reasoning(text: str) -> list[str]:
    raw_lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if not raw_lines:
        return []
    kept: list[str] = []
    used = 0
    for line in raw_lines:
        if len(kept) >= _REASONING_MAX_LINES:
            break
        remaining = _REASONING_MAX_CHARS - used
        if remaining <= 0:
            break
        if len(line) > remaining:
            kept.append(line[: max(remaining - 1, 1)] + "…")
            used = _REASONING_MAX_CHARS
            break
        kept.append(line)
        used += len(line)
    if len(raw_lines) > len(kept) or used >= _REASONING_MAX_CHARS:
        if kept:
            kept[-1] = kept[-1].rstrip("…") + "…"
    return kept


def unpack_assistant_pending(item: object) -> tuple[str, TurnTrace | None]:
    if isinstance(item, tuple) and item:
        text = str(item[0])
        trace = item[1] if len(item) > 1 else None
        if isinstance(trace, TurnTrace):
            return text, trace
        return text, None
    return str(item), None
