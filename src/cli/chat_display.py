from __future__ import annotations

import datetime as _dt
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.markup import escape as _escape_markup

from backend.chat_resolve import is_server_backend
from backend.hf.registry import HF_MODEL_PRESETS

if TYPE_CHECKING:
    from cli.chat import _SessionState


_BLUE = "#60a5fa"
_BLUE_DIM = "#93c5fd"
_BLUE_DEEP = "#3b82f6"

ESSENTIAL_SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "full command list (also ? or F1 in TUI)"),
    ("/quit", "exit chat (q or Ctrl+Q in TUI)"),
    ("/speak TEXT", "speak now (Pipecat Kokoro default)"),
    ("/transcribe PATH", "transcribe audio with Qwen ASR"),
    ("/listen [S]", "mic into prompt (Ctrl+L in TUI)"),
    ("/tts on|off", "auto-speak assistant replies"),
    ("/backend", "ollama | lmstudio | hf (auto on startup)"),
    ("/models", "open model picker (TUI) or list sources"),
    ("/model PRESET", "switch model (or server model id)"),
    ("/params", "show sampling / max tokens (change with /temp /max …)"),
    ("/tool-rounds", "tool-loop rounds (N or unlimited); /unlimited to uncap"),
    ("/model-download PRESET", "fetch weights from Hugging Face Hub"),
    ("/reset", "clear conversation, keep system prompt"),
    ("/save [path]", "export transcript"),
    ("/tokens", "prompt tokens vs context window"),
    ("/regen", "redo last reply (r)"),
    ("/rag-status", "retrieval backend status"),
    ("/tools", "show enabled speak / vault / shell tools"),
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


def _spread(left: str, right: str, *, width: int) -> str:
    left = left.rstrip()
    right = right.strip()
    if not right:
        return _fit(left, width)
    if len(left) + 1 + len(right) > width:
        budget = max(width - len(right) - 1, 8)
        left = _fit(left, budget)
    gap = width - len(left) - len(right)
    if gap < 1:
        return _fit(f"{left} {right}", width)
    return left + (" " * gap) + right


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


def _model_identity(state: _SessionState) -> str:
    if is_server_backend(state.backend_id):
        model = state.server_model or "(none)"
        return f"{state.backend_id} · {model}"
    if state.preset_key:
        return f"hf · {state.preset_key}"
    if state.model_path.strip():
        name = state.model_path.replace("\\", "/").rstrip("/").split("/")[-1]
        return f"hf · {name or state.model_path}"
    return f"{state.backend_id} · (no model)"


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


def _status_bits(state: _SessionState) -> str:
    from integrations.obsidian.client import obsidian_tools_enabled
    from integrations.shell.runner import shell_tools_enabled

    bits: list[str] = []
    if state.retriever is not None:
        bits.append(f"rag {state.retriever.backend_id()}")
    bits.append(f"tts {'on' if state.tts_enabled else 'off'}")
    bits.append(f"sst {'on' if state.sst_enabled else 'off'}")
    bits.append(f"obsidian {'on' if obsidian_tools_enabled() else 'off'}")
    bits.append(f"shell {'on' if shell_tools_enabled() else 'off'}")
    if state.memory is not None and state.memory_scope is not None:
        bits.append(f"mem {state.memory_scope.session_id}")
    return " · ".join(bits)


def _tools_status_bits() -> str:
    from integrations.obsidian.client import obsidian_tools_enabled
    from integrations.shell.runner import shell_tools_enabled

    return " · ".join(
        [
            f"obsidian {'on' if obsidian_tools_enabled() else 'off'}",
            f"shell {'on' if shell_tools_enabled() else 'off'}",
        ]
    )


def chat_ready_status_line(state: _SessionState) -> str:
    _ = state
    return "Ready · /models or Ctrl+M · Ctrl+D/W/E/G panes · F1 = /commands"


def _runtime_stats_summary(state: _SessionState) -> str:
    return state.stats.format_banner_stats()


def chat_banner_lines(
    state: _SessionState,
    *,
    log_path: str | None = None,
    started_at: _dt.datetime | None = None,
    for_markup: bool = False,
    width: int | None = None,
) -> list[str]:
    """Plain-text startup banner for REPL and TUI copy buffer."""
    return [
        line.plain
        for line in chat_banner_transcript_lines(
            state,
            log_path=log_path,
            started_at=started_at,
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
    from cli.chat import chat_session_ready

    _ = log_path
    _ = started_at
    w = resolve_banner_width(width)
    lines: list[TranscriptLine] = []

    def add(plain: str, markup: str | None = None) -> None:
        plain = _fit(plain, w)
        lines.append(TranscriptLine(plain=plain, markup=markup if for_markup and markup else plain))

    def add_frame(plain: str, *, emphasis: bool = False) -> None:
        color = _BLUE_DEEP if emphasis else _BLUE
        add(plain, f"[bold {color}]{_escape_markup(plain)}[/]")

    right_meta_parts = [p for p in (_gen_params_summary(state), _context_summary(state)) if p]
    identity = _spread(
        f" {_model_identity(state)}",
        f"{'  '.join(right_meta_parts)} " if right_meta_parts else "",
        width=max(w - 2, 1),
    )
    cmds = _fit(" /help · /quit · /tools · /speak · /listen · /transcribe · /tts · /sst · /backend · /models · /params · F1", max(w - 2, 1))
    status_left = _status_bits(state)
    status_right = _runtime_stats_summary(state) or "Ctrl+M models"
    status = _spread(
        f" {status_left}" if status_left else " ",
        f"{status_right} ",
        width=max(w - 2, 1),
    )

    add_frame(_box_outer(w, top=True), emphasis=True)
    add_frame(_box_center("", w))
    add_frame(_box_center("ORODRUIN  CHAT", w), emphasis=True)
    add_frame(_box_center("", w))
    add_frame(_box_rule(w), emphasis=True)

    if not chat_session_ready(state):
        if is_server_backend(state.backend_id):
            warn = f" no {state.backend_id} model - /models"
        else:
            warn = " no weights - /model-download or /models"
        row = _box_row(warn, w)
        add(row, f"[yellow]{_escape_markup(row)}[/]")

    row = _box_row(identity, w)
    add(row, f"[bold]{_escape_markup(row)}[/]")
    row = _box_row(cmds, w)
    add(row, f"[{_BLUE_DIM}]{_escape_markup(row)}[/]")
    row = _box_row(status, w)
    add(row, f"[dim]{_escape_markup(row)}[/]")
    add_frame(_box_outer(w, top=False), emphasis=True)
    add("")
    return lines


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
) -> TranscriptLine:
    ts = (timestamp or _dt.datetime.now()).strftime("%H:%M:%S")
    plain = f"[{ts}] turn {turn:>3}  assistant>"
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    markup = f"[dim]{ts}[/] [dim]turn {turn:>3}[/]  [bold {_BLUE_DIM}]assistant>[/]"
    return TranscriptLine(plain=plain, markup=markup)


def format_assistant_body_line(text: str, *, for_markup: bool = False) -> TranscriptLine:
    plain = text
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    return TranscriptLine(plain=plain, markup=f"  {_escape_markup(text)}")


def format_system_line(text: str, *, for_markup: bool = False) -> TranscriptLine:
    plain = text
    if not for_markup:
        return TranscriptLine(plain=plain, markup=plain)
    if text.startswith("(") and text.endswith(")"):
        return TranscriptLine(plain=plain, markup=f"[dim italic]{_escape_markup(text)}[/]")
    return TranscriptLine(plain=plain, markup=f"[dim]{_escape_markup(text)}[/]")
