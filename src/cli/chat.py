from __future__ import annotations

import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

# Todo: remove this path bootstrap after the package uses consistent sophon.* imports end-to-end.

import datetime as _dt
import json
import logging
import os
import signal
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Literal

from cli.chat_stats import SessionStats, initial_debug_mode_from_env
from cli.chat_trace import (
    HeartbeatStep,
    RetrievalTrace,
    ToolCallTrace,
    TurnTrace,
    args_preview,
    build_turn_trace,
    format_heartbeat_step_plain,
    tool_api_label,
)
from backend.hf.backend import (
    GenerationResult,
    ModelMeta,
    generate_response,
    load_processor_and_model,
    parsed_to_display_text,
    read_model_meta,
)
from backend.chat_resolve import (
    ChatBackendId,
    backend_help_tokens,
    chat_backend_ids,
    is_managed_backend,
    is_server_backend,
    list_server_models,
    lmstudio_reachable,
    managed_backend_ids,
    normalize_chat_backend_token,
    ollama_reachable,
    provider_api_key,
    provider_key_hint,
    missing_provider_key_message,
    resolve_chat_backend,
    server_backend_reachable,
    server_chat_complete,
    server_target_prefixes,
    sync_managed_model_index,
)

from backend.hf.paths import (
    infer_default_quantization,
    require_model_on_disk,
    resolve_cli_quantization,
    resolve_local_model_dir,
)
from processing.text.context import ContextBuildResult, build_messages_for_model
from processing.text.memory import (
    MemoryBudget,
    MemoryLayer,
    MemoryScope,
    MemoryStore,
    budget_from_env,
    open_memory_layer,
    open_memory_store,
)
from processing.text.retrieval import RagRetriever, RetrievalQuery, RetrievalResult, load_rag_retriever, rag_retriever_ids
from processing.text.skills import SkillCatalog, open_skill_catalog
from processing.text.retrieval.adaptive import AdaptiveDecision, RetrievalDecision, decide_retrieval
from processing.text.retrieval.types import empty_result
from processing.audio.speech.cli import SttCliOptions, TtsCliOptions
from processing.audio.speech.factory import create_speech_stt_engine, create_speech_tts_engine
from processing.audio.speech.protocols import SpeechSttEngine, SpeechTtsEngine
from processing.audio.speech.playback import play_wav_file
from processing.audio.speech.session_store import (
    SessionSpeechStore,
    SpeechClip,
    default_replay_speeds,
    format_replay_speed,
    parse_replay_speed,
)
from processing.audio.speech.speak import speak_text_blocking, speak_text_to_store
from processing.audio.speech.types import (
    TtsBackendId,
    SttBackendId,
    normalize_stt_backend_token,
    normalize_stt_language,
    normalize_tts_backend_token,
    stt_backend_help_tokens,
    tts_backend_help_tokens,
)
from utils.device.env_bootstrap import sophon_chat_logs_dir, sophon_project_root
from backend.hf.registry import (
    HF_MODEL_PRESETS,
    local_only_model_dirs,
    match_preset_key,
    model_dir_has_complete_weights,
    preset_has_weights,
    preset_keys_sorted,
    resolve_chat_startup_model,
    resolve_preset_dir,
)
from utils.download.hf import (
    capture_hub_download_logs,
    capture_hub_user_warnings,
    configure_hub_verbose,
    download_preset_snapshot,
    format_download_progress,
    hub_tqdm_bridge_factory,
    is_file_count_progress,
    parse_hub_size_hint,
    scan_local_dir_download_bytes,
    split_hub_progress_label,
    throttled_progress_callback,
    watch_local_download_bytes,
)


@dataclass
class _GenParams:
    max_new_tokens: int = 2048
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None


@dataclass
class ChatCliParams:
    model: str | None
    preset: str | None
    quantization: str
    qbit: int | None
    system: str | None
    max_new_tokens: int
    thinking: bool | None
    raw: bool
    debug: bool | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    seed: int | None
    rag: str
    rag_index: str | None
    rag_top_k: int
    rag_adaptive: bool
    rag_structure: str
    rag_structure_dir: str | None
    memory_db: str | None
    memory_session: str | None
    memory_user: str | None
    memory_recall_turns: int
    tts: TtsCliOptions
    sst: SttCliOptions
    chat_first: bool = True
    startup_preload: bool = False


@dataclass(frozen=True)
class ChatIo:
    emit: Callable[[str], None]
    on_assistant: Callable[..., None]
    on_dictate: Callable[[str], None] | None = None
    on_speech_clip: Callable[[SpeechClip], None] | None = None
    on_mic: Callable[[str], None] | None = None
    on_clear: Callable[[], None] | None = None
    on_step: Callable[[HeartbeatStep], None] | None = None
    on_heartbeat_end: Callable[[], None] | None = None
    on_permission: Callable[[object], str] | None = None
    on_setup_choice: Callable[[str], str] | None = None


def _default_emit(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _default_on_assistant(text: str, trace: TurnTrace | None = None) -> None:
    if trace is not None:
        from cli.chat_display import format_turn_trace_lines

        for line in format_turn_trace_lines(trace, for_markup=False, reply_text=text):
            print(line.plain, file=sys.stderr, flush=True)
    print(text)
    print(flush=True)


def _default_on_clear() -> None:
    return


def _default_on_step(step: HeartbeatStep) -> None:
    print(format_heartbeat_step_plain(step), file=sys.stderr, flush=True)


def _default_on_heartbeat_end() -> None:
    return


def _default_on_permission(request: object) -> str:
    from harness.approval import DECISION_DENY, PermissionRequest
    from harness.prompt import prompt_stdio

    if not isinstance(request, PermissionRequest):
        return DECISION_DENY
    return prompt_stdio(request)


_DEFAULT_IO = ChatIo(
    emit=_default_emit,
    on_assistant=_default_on_assistant,
    on_clear=_default_on_clear,
    on_step=_default_on_step,
    on_heartbeat_end=_default_on_heartbeat_end,
    on_permission=_default_on_permission,
)
_CURRENT_IO = _DEFAULT_IO


def configure_chat_io(io: ChatIo) -> None:
    global _CURRENT_IO
    _CURRENT_IO = io


def reset_chat_io() -> None:
    global _CURRENT_IO
    _CURRENT_IO = _DEFAULT_IO


def _wipe_transcript_ui() -> None:
    cb = _CURRENT_IO.on_clear
    if cb is not None:
        cb()


def _emit_step(step: HeartbeatStep) -> None:
    cb = _CURRENT_IO.on_step
    if cb is not None:
        cb(step)
        return
    _default_on_step(step)


def _ask_permission(request: object) -> str:
    from harness.approval import DECISION_DENY, PermissionRequest

    name = getattr(request, "tool", "tool") if request is not None else "tool"
    _emit_step(
        HeartbeatStep(kind="permission", name=str(name), preview="waiting 1/2/3", ok=True)
    )
    cb = _CURRENT_IO.on_permission
    if cb is None:
        return DECISION_DENY
    if not isinstance(request, PermissionRequest):
        return DECISION_DENY
    # TODO: abort the wait when the user hits Esc / stop on the generation turn
    return cb(request)


def _emit_think(thought: str) -> None:
    body = (thought or "").strip()
    if not body:
        return
    preview = body if len(body) <= 2400 else body[:2400] + "…"
    _emit_step(HeartbeatStep(kind="think", name="reasoning", preview=preview, ok=True))


def _emit_heartbeat_end() -> None:
    cb = _CURRENT_IO.on_heartbeat_end
    if cb is not None:
        cb()


def _tool_progress_line(name: str, preview: object) -> None:
    if _CURRENT_IO.on_step is not None:
        return
    extra = str(preview or "").strip()
    _emit(f"({name} {extra})" if extra else f"({name})")


def _arg_path(args: dict) -> str | None:
    for key in ("path", "file"):
        raw = args.get(key)
        if raw:
            return str(raw).strip()
    return None


def _emit_tool_heartbeat(name: str, args: dict, result: str, latency_s: float) -> None:
    preview = args_preview(name, args)
    ok = not str(result).lower().startswith("error:")
    path = _arg_path(args)
    _emit_step(
        HeartbeatStep(
            kind="tool",
            name=name or "unknown",
            preview=preview,
            path=path,
            ok=ok,
            latency_s=latency_s,
        )
    )
    if str(name).startswith("skill_"):
        skill_name = str(args.get("name") or "").strip()
        _emit_step(
            HeartbeatStep(
                kind="skill",
                name=name,
                preview=skill_name or preview,
                path=path,
                ok=ok,
                latency_s=latency_s,
            )
        )
    if path:
        _emit_step(
            HeartbeatStep(
                kind="file",
                name=name,
                preview=path,
                path=path,
                ok=ok,
                latency_s=latency_s,
            )
        )


def _emit_context_heartbeat(state: _SessionState, built: object) -> None:
    attached = ", ".join(state.attached_skills) if state.attached_skills else "none"
    dropped = state.last_memory_dropped
    blocks = getattr(built, "injected_blocks", None) or []
    block_s = ",".join(str(b) for b in blocks) if blocks else "none"
    _emit_step(
        HeartbeatStep(
            kind="context",
            name="pack",
            preview=f"attached={attached} dropped={dropped} blocks={block_s}",
            ok=True,
        )
    )
    _emit_step(
        HeartbeatStep(kind="idle", name="model", preview="waiting", ok=True)
    )


def _emit_assistant(text: str, trace: TurnTrace | None = None) -> None:
    cb = _CURRENT_IO.on_assistant
    if trace is not None:
        try:
            cb(text, trace)
            return
        except TypeError:
            pass
    cb(text)


@dataclass
class _SessionState:
    processor: object | None
    model: object | None
    meta: ModelMeta | None
    messages: list[dict[str, object]] = field(default_factory=list)
    system_text: str | None = None
    enable_thinking: bool = False
    strip: bool = True
    debug: bool = False
    stats: SessionStats = field(default_factory=SessionStats)
    params: _GenParams = field(default_factory=_GenParams)
    exit_requested: bool = False
    retriever: RagRetriever | None = None
    structure_retriever: RagRetriever | None = None
    retrieval_top_k: int = 5
    rag_adaptive: bool = True
    last_retrieval_decision: dict | None = None
    memory: MemoryStore | None = None
    memory_scope: MemoryScope | None = None
    memory_recall_turns: int = 6
    memory_layer: MemoryLayer | None = None
    memory_budget: MemoryBudget | None = None
    last_memory_dropped: int = 0
    skill_catalog: SkillCatalog | None = None
    attached_skills: list[str] = field(default_factory=list)
    _last_persisted_user_obj_id: int = 0
    tts_enabled: bool = False
    tts_plain_text: bool = True
    tts_model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    tts_speaker: str = "am_adam"
    tts_language: str = "English"
    tts_instruct: str | None = None
    tts_max_chars: int = 8000
    tts_device: str | None = None
    tts_backend_id: TtsBackendId = "pipecat"
    tts_ollama_model: str | None = None
    tts_speed: float = 1.0
    tts_engine: SpeechTtsEngine | None = None
    sst_enabled: bool = False
    sst_backend_id: SttBackendId = "hf_qwen"
    sst_model_id: str = "Qwen/Qwen3-ASR-0.6B"
    sst_language: str | None = None
    sst_device: str | None = None
    sst_max_new_tokens: int = 1024
    sst_engine: SpeechSttEngine | None = None
    sst_mic: object | None = None
    sst_busy: bool = False
    model_path: str = ""
    preset_key: str | None = None
    quantization: str = "none"
    project_root: Path = field(default_factory=Path.cwd)
    model_switching: bool = False
    adapter_path: str | None = None
    last_finetune_run_dir: Path | None = None
    last_finetune_adapter_name: str | None = None
    train_loss_history: list[float] = field(default_factory=list)
    finetune_running: bool = False
    last_eval_summary: dict | None = None
    eval_running: bool = False
    backend_id: ChatBackendId = "hf"
    server_model: str | None = None
    last_retrieval_metrics: dict | None = None
    retrieval_probe_history: list = field(default_factory=list)
    shell_session: object | None = None
    tool_max_rounds: int | None = None
    assist: object | None = None
    editor_open_path: str | None = None
    editor_open_text: str | None = None
    editor_workspace: Path | None = None
    assist_ui_hook: object | None = None
    speech_store: SessionSpeechStore | None = None
    harness: object | None = None
    last_permission: str = ""
    shell_mode: bool = False
    auto_attach: bool = True
    slash_recents: list[str] = field(default_factory=list)
    trainer_driver: str = "hf-lora"
    trainer_task: str = ""
    trainer_data: str = ""
    trainer_model: str = ""
    trainer_target: str = ""
    trainer_fast_dev: bool = True
    ssl4sed_running: bool = False
    ssl4sed_proc: object | None = None
    ssl4sed_thread: object | None = None
    ssl4sed_run_dir: str = ""
    ssl4sed_run_meta: str = ""
    ssl4sed_run_log: str = ""
    ssl4sed_default_target: str = ""
    ssl4sed_registry: list | None = None
    setup_pending: dict | None = None
    setup_last_table: str = ""
    setup_last_recipe: str = ""
    delegation_depth: int = 0
    quiet: bool = False
    subagent_agent_type: str = "general"
    subagent_service: object | None = None
    last_subagent_run: object | None = None


def _format_tool_max_rounds(value: int | None) -> str:
    if value is None:
        return "unlimited"
    return str(value)


def _tool_max_rounds_from_env() -> int | None:
    raw = os.environ.get("SOPHON_TOOL_MAX_ROUNDS", "").strip().lower()
    if raw in ("", "unlimited", "inf", "infinite", "none", "0"):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _thinking_default(cli_flag: bool | None) -> bool:
    if cli_flag is not None:
        return cli_flag
    return os.environ.get("GEMMA4_THINKING", "").lower() in ("1", "true", "yes")


def _baseline_messages(system_text: str | None) -> list[dict[str, object]]:
    if system_text is None or str(system_text).strip() == "":
        return []
    return [{"role": "system", "content": str(system_text).strip()}]


def _suppress_noisy_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*max_new_tokens.*max_length.*",
    )
    warnings.filterwarnings("ignore", message=r".*triton not found.*")
    for name in (
        "transformers.generation.configuration_utils",
        "transformers.generation.utils",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _read_user_input(prompt: str) -> str | None:
    try:
        line = input(prompt)
    except EOFError:
        return None
    if "\"\"\"" not in line:
        return line
    if line.count("\"\"\"") >= 2:
        return line
    lines = [line]
    try:
        while True:
            extra = input("... ")
            lines.append(extra)
            if "\"\"\"" in extra:
                break
    except EOFError:
        pass
    return "\n".join(lines)


@dataclass
class _Command:
    name: str
    help_text: str
    handler: Callable[["_SessionState", str], bool]


_COMMANDS: dict[str, _Command] = {}
_SHORTCUTS: dict[str, str] = {}

# Todo: add each new slash command name to _HELP_SECTIONS. Unlisted names render under other.
_HELP_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "session",
        ("help", "quit", "reset", "clear", "save", "log", "pop", "regen", "system", "tokens", "stats", "debug", "chat", "auto-attach"),
    ),
    (
        "model",
        ("backend", "models", "models-sync", "model", "model-download", "setup"),
    ),
    (
        "generation",
        ("params", "temp", "top-p", "top-k", "max", "seed", "rep", "tool-rounds", "unlimited"),
    ),
    (
        "tools",
        (
            "tools",
            "mode",
            "permissions",
            "obsidian-status",
            "zotero-status",
            "zotero-tree",
            "zotero-search",
            "zotero-metrics",
            "zotero-read",
            "google-status",
            "overleaf-status",
            "overleaf-list",
            "overleaf-read",
            "assist",
        ),
    ),
    (
        "subagents",
        ("subagents",),
    ),
    (
        "speech",
        (
            "speak",
            "speak-file",
            "speak-last",
            "tts",
            "tts-test",
            "tts-plain",
            "tts-speaker",
            "tts-speed",
            "tts-lang",
            "tts-instruct",
            "tts-max-chars",
            "tts-backend",
            "tts-model",
            "tts-ollama-model",
            "transcribe",
            "listen",
            "listen-cancel",
            "play",
            "clips",
            "sst",
            "sst-backend",
            "sst-model",
            "sst-lang",
        ),
    ),
    (
        "memory",
        ("memory", "memory-status", "memory-clear", "rag-status", "rag-probe", "rag-index"),
    ),
    (
        "skills",
        ("skill",),
    ),
    (
        "eval",
        ("eval",),
    ),
    (
        "train",
        ("finetune", "finetune-status", "finetune-datasets", "finetune-config", "adapter", "trainer"),
    ),
)


def _register(
    name: str,
    help_text: str,
    shortcut: str | None = None,
    aliases: tuple[str, ...] = (),
):
    def deco(fn: Callable[["_SessionState", str], bool]) -> Callable[["_SessionState", str], bool]:
        cmd = _Command(name=name, help_text=help_text, handler=fn)
        _COMMANDS[name] = cmd
        for alias in aliases:
            _COMMANDS[alias] = cmd
        if shortcut:
            _SHORTCUTS[shortcut] = name
        return fn

    return deco


def _emit(msg: str) -> None:
    _CURRENT_IO.emit(msg)


def _dictate(text: str) -> None:
    body = text.strip()
    if not body:
        return
    if _CURRENT_IO.on_dictate is not None:
        _CURRENT_IO.on_dictate(body)
        return
    _emit(body)


def _announce_speech_clip(clip: SpeechClip) -> None:
    cb = _CURRENT_IO.on_speech_clip
    if cb is not None:
        cb(clip)
        return
    from cli.chat_display import format_speech_clip_line

    _emit(format_speech_clip_line(clip).plain)


def _notify_mic(phase: str) -> None:
    cb = _CURRENT_IO.on_mic
    if cb is not None:
        cb(phase)


def _alias_names() -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for key, cmd in _COMMANDS.items():
        if key != cmd.name:
            aliases.setdefault(cmd.name, []).append(key)
    return aliases


def _format_help_command(
    cmd: _Command,
    shortcut_for: dict[str, str],
    aliases: dict[str, list[str]] | None = None,
) -> str:
    extra: list[str] = []
    sc = shortcut_for.get(cmd.name)
    if sc:
        extra.append(sc)
    for alias in (aliases or {}).get(cmd.name, []):
        extra.append(alias)
    suffix = " | ".join(extra)
    head = f"  /{cmd.name}" + (f" | {suffix}" if suffix else "")
    return f"{head:<24} {cmd.help_text}"


def _format_help(section: str | None = None) -> str:
    shortcut_for = {v: k for k, v in _SHORTCUTS.items()}
    aliases = _alias_names()
    listed: set[str] = set()
    listed_names: set[str] = set()
    blocks: list[tuple[str, list[str]]] = []
    for name, cmd_names in _HELP_SECTIONS:
        lines: list[str] = []
        for cmd_name in cmd_names:
            cmd = _COMMANDS.get(cmd_name)
            if cmd is None:
                continue
            listed.add(cmd_name)
            listed_names.add(cmd.name)
            lines.append(_format_help_command(cmd, shortcut_for, aliases))
        if lines:
            blocks.append((name, lines))
    other_seen: set[str] = set()
    other_lines: list[str] = []
    for cmd_name, cmd in _COMMANDS.items():
        if cmd_name in listed or cmd.name in listed_names or cmd.name in other_seen:
            continue
        other_seen.add(cmd.name)
        other_lines.append(_format_help_command(cmd, shortcut_for, aliases))
    if other_lines:
        blocks.append(("other", other_lines))

    wanted = (section or "").strip().lower()
    if wanted:
        match = [(name, lines) for name, lines in blocks if name == wanted]
        if not match:
            available = " ".join(name for name, _lines in blocks)
            return f"unknown help section {wanted!r}. sections: {available}"
        blocks = match

    rows: list[str] = ["commands:" if not wanted else f"commands ({wanted}):"]
    for name, lines in blocks:
        rows.append("")
        rows.append(f"{name}:")
        rows.extend(lines)
    rows.append("")
    rows.append(
        "  /help [section]          Filter by section "
        "(session model generation tools subagents speech memory skills eval train)."
    )
    rows.append("  \"\"\"multi-line\"\"\"        Triple-quoted input is gathered until the closing \"\"\".")
    return "\n".join(rows)


format_chat_help = _format_help


@dataclass(frozen=True)
class SlashCommandInfo:
    name: str
    help_text: str
    section: str
    aliases: tuple[str, ...]
    shortcut: str


def list_slash_commands() -> tuple[SlashCommandInfo, ...]:
    section_of: dict[str, str] = {}
    for section, names in _HELP_SECTIONS:
        for cmd_name in names:
            section_of.setdefault(cmd_name, section)
    shortcut_for = {v: k for k, v in _SHORTCUTS.items()}
    aliases = _alias_names()
    seen: set[str] = set()
    rows: list[SlashCommandInfo] = []
    ordered_names: list[str] = []
    for _section, names in _HELP_SECTIONS:
        for cmd_name in names:
            if cmd_name not in ordered_names:
                ordered_names.append(cmd_name)
    for cmd_name, cmd in _COMMANDS.items():
        if cmd.name not in ordered_names:
            ordered_names.append(cmd.name)
    for cmd_name in ordered_names:
        cmd = _COMMANDS.get(cmd_name)
        if cmd is None or cmd.name in seen:
            continue
        seen.add(cmd.name)
        section = section_of.get(cmd.name) or section_of.get(cmd_name) or "other"
        alias_list = tuple(sorted(aliases.get(cmd.name, [])))
        rows.append(
            SlashCommandInfo(
                name=cmd.name,
                help_text=cmd.help_text,
                section=section,
                aliases=alias_list,
                shortcut=shortcut_for.get(cmd.name, ""),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class ModelPickerEntry:
    target: str
    title: str
    detail: str
    local: bool
    current: bool
    kind: Literal["header", "preset", "path", "adapter"] = "preset"
    group: str = ""
    child_count: int = 0


def chat_session_has_weights(state: _SessionState) -> bool:
    return state.processor is not None and state.model is not None and state.meta is not None


def chat_session_ready(state: _SessionState) -> bool:
    if is_server_backend(state.backend_id):
        return bool(state.server_model and str(state.server_model).strip())
    return chat_session_has_weights(state)


def harness_inventory_counts(state: _SessionState) -> tuple[int, int, int]:
    from cli.chat_tools import default_chat_tools

    catalog = state.skill_catalog
    n_skills = len(catalog.entries) if catalog is not None else 0
    tools = default_chat_tools(
        tts_tool=_tts_tools_enabled(),
        sst_tool=_sst_tools_enabled(),
        obsidian_tool=_obsidian_tools_wanted(),
        zotero_tool=_zotero_tools_wanted(),
        google_tool=_google_tools_wanted(),
        overleaf_tool=_overleaf_tools_wanted(),
        shell_tool=_shell_tools_wanted(),
        editor_tool=_editor_tools_wanted(),
        memory_tool=state.memory_layer is not None and _memory_tools_wanted(),
        skill_tool=catalog is not None and _skill_tools_wanted(),
        subagent_tool=_spawn_tools_in_schema(state),
    )
    n_mcp = 0
    # TODO: count live MCP connectors once a connector registry exists
    return n_skills, len(tools), n_mcp


def _env_key_set_hint(key: str) -> str:
    from utils.device.env_bootstrap import dotenv_location_label

    return f"(set {key} in {dotenv_location_label()})"


def _colon_provider_hint(raw: str) -> str | None:
    token = raw.strip()
    if ":" in token or "/" in token or "\\" in token or "_" not in token:
        return None
    head, _rest = token.split("_", 1)
    aliases = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google",
        "gemini": "google",
        "ollama": "ollama",
        "lmstudio": "lmstudio",
    }
    backend = aliases.get(head.lower())
    if backend is None:
        return None
    return (
        f"Server models use a colon prefix, not underscore. "
        f"Example: openai:gpt-4.1 (not openai_gpt4.1)"
    )


def _content_to_text(content: object) -> str:
    from cli.agent_runtime import content_to_text

    return content_to_text(content)


def _messages_for_server(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    from cli.agent_runtime import messages_for_server

    return messages_for_server(messages)


def _pick_default_server_model(backend_id: ChatBackendId) -> str | None:
    try:
        names = list_server_models(backend_id)
    except Exception:
        return None
    return names[0] if names else None


def _parse_server_target(token: str) -> tuple[ChatBackendId, str] | None:
    raw = token.strip()
    lower = raw.lower()
    for prefix, backend in server_target_prefixes():
        if lower.startswith(prefix):
            name = raw[len(prefix) :].strip()
            if name:
                return backend, name
    return None


def _is_current_server_entry(state: _SessionState, backend: ChatBackendId, name: str) -> bool:
    return state.backend_id == backend and state.server_model == name


def _append_server_section(
    entries: list[ModelPickerEntry],
    state: _SessionState,
    *,
    backend: ChatBackendId,
    reachable: bool,
    missing_key: str | None = None,
) -> None:
    from backend.model_index import load_provider_snapshot

    group = backend
    names: list[str] = []
    detail = ""
    if missing_key and not is_managed_backend(backend):
        entries.append(
            ModelPickerEntry("", backend, _env_key_set_hint(missing_key), False, False, "header", group, 0)
        )
        return
    if is_managed_backend(backend):
        names = list_server_models(backend)
        _, fetched_at = load_provider_snapshot(backend)
        if fetched_at is None and not names:
            if missing_key:
                detail = _env_key_set_hint(missing_key)
            else:
                detail = "/models-sync"
        elif missing_key:
            detail = f"{len(names)} · key missing"
        else:
            detail = str(len(names))
    elif not reachable:
        entries.append(
            ModelPickerEntry("", backend, "unreachable", False, False, "header", group, 0)
        )
        return
    else:
        try:
            names = list_server_models(backend)
        except Exception as exc:
            entries.append(
                ModelPickerEntry("", backend, f"list failed: {exc}", False, False, "header", group, 0)
            )
            return
        detail = str(len(names)) if names else "no models exposed"
    entries.append(
        ModelPickerEntry("", backend, detail, False, False, "header", group, len(names))
    )
    for name in names:
        entries.append(
            ModelPickerEntry(
                target=f"{backend}:{name}",
                title=name,
                detail=backend,
                local=True,
                current=_is_current_server_entry(state, backend, name),
                kind="preset",
                group=group,
            )
        )


def switch_to_server_model(state: _SessionState, backend: ChatBackendId, name: str) -> None:
    if is_managed_backend(backend) and not provider_api_key(backend):
        _emit(missing_provider_key_message(backend))
        return
    try:
        names = list_server_models(backend)
    except Exception as exc:
        _emit(f"(could not list {backend} models: {exc})")
        return
    match = None
    lower = name.lower()
    for item in names:
        if item == name or item.lower() == lower:
            match = item
            break
    if match is None:
        for item in names:
            if lower in item.lower():
                match = item
                break
    if match is None and is_managed_backend(backend) and name.strip():
        match = name.strip()
        _emit(f"(not in local index for {backend}; using {match!r}. /models-sync to refresh)")
    if match is None:
        _emit(f"unknown {backend} model: {name!r}. Use /models.")
        return
    if state.backend_id == backend and match == state.server_model:
        _emit(f"(already using {backend}:{match})")
        return
    if state.backend_id != backend or not is_server_backend(state.backend_id):
        _unload_model_weights(state)
    state.backend_id = backend
    state.server_model = match
    state.model_path = f"{backend}:{match}"
    state.preset_key = None
    state.meta = None
    state.messages = _baseline_messages(state.system_text)
    state._last_persisted_user_obj_id = 0
    state.stats = SessionStats(max_position_embeddings=None)
    _emit(f"(switched to {backend}:{match}; conversation cleared)")


def switch_server_model(state: _SessionState, target: str) -> None:
    parsed = _parse_server_target(target)
    if parsed is not None:
        switch_to_server_model(state, parsed[0], parsed[1])
        return
    switch_to_server_model(state, state.backend_id, target)


def switch_session_backend(state: _SessionState, token: str) -> None:
    backend = normalize_chat_backend_token(token)
    if backend is None:
        _emit(f"usage: /backend [{' | '.join(('auto',) + chat_backend_ids())}]")
        return
    if backend == state.backend_id:
        _emit(f"(already on backend {backend})")
        return
    if is_managed_backend(backend) and not provider_api_key(backend):
        _emit(missing_provider_key_message(backend))
        return
    _unload_model_weights(state)
    state.backend_id = backend
    state.server_model = None
    state.preset_key = None
    state.model_path = ""
    state.meta = None
    state.messages = _baseline_messages(state.system_text)
    state._last_persisted_user_obj_id = 0
    state.stats = SessionStats(max_position_embeddings=None)
    if is_server_backend(backend):
        picked = _pick_default_server_model(backend)
        if picked:
            state.server_model = picked
            state.model_path = f"{backend}:{picked}"
            _emit(f"(backend={backend}; model={picked})")
        else:
            _emit(f"(backend={backend}; no models listed — pick one in /models)")
        return
    _emit("(backend=hf; use /models and /model PRESET to load weights)")


def build_model_picker_entries_for_root(
    root: Path,
    *,
    current_preset: str | None = None,
    current_path: Path | None = None,
    current_backend: str = "hf",
) -> list[ModelPickerEntry]:
    downloaded = [key for key in preset_keys_sorted() if preset_has_weights(key, root)]
    missing = [key for key in preset_keys_sorted() if not preset_has_weights(key, root)]
    extra_dirs = local_only_model_dirs(root)

    entries: list[ModelPickerEntry] = []
    if downloaded or extra_dirs:
        local_count = len(downloaded) + len(extra_dirs)
        entries.append(
            ModelPickerEntry(
                "",
                "huggingface (local)",
                str(local_count),
                False,
                False,
                "header",
                "hf-local",
                local_count,
            )
        )
        for key in downloaded:
            preset = HF_MODEL_PRESETS[key]
            is_current = current_backend == "hf" and (
                key == current_preset
                or (
                    current_path is not None
                    and resolve_preset_dir(key, root).resolve() == current_path
                )
            )
            entries.append(
                ModelPickerEntry(
                    target=f"hf:{key}",
                    title=key,
                    detail=preset.repo_id,
                    local=True,
                    current=is_current,
                    kind="preset",
                    group="hf-local",
                )
            )
        for path in extra_dirs:
            is_current = (
                current_backend == "hf"
                and current_path is not None
                and path.resolve() == current_path
            )
            entries.append(
                ModelPickerEntry(
                    target=str(path),
                    title=path.name,
                    detail=str(path),
                    local=True,
                    current=is_current,
                    kind="path",
                    group="hf-local",
                )
            )

    if missing:
        entries.append(
            ModelPickerEntry(
                "",
                "huggingface (not downloaded)",
                str(len(missing)),
                False,
                False,
                "header",
                "hf-missing",
                len(missing),
            )
        )
        for key in missing:
            preset = HF_MODEL_PRESETS[key]
            entries.append(
                ModelPickerEntry(
                    target=f"hf:{key}",
                    title=key,
                    detail=preset.repo_id,
                    local=False,
                    current=False,
                    kind="preset",
                    group="hf-missing",
                )
            )
    return entries


def _server_path_prefixes() -> tuple[str, ...]:
    return tuple(prefix for prefix, _backend in server_target_prefixes()) + ("hf:",)


def _managed_picker_state(backend: ChatBackendId) -> tuple[bool, str | None]:
    if not provider_api_key(backend):
        return False, provider_key_hint(backend)
    return True, None


def _adapter_index_entries(project_root: Path) -> list:
    try:
        from training.common.index import load_adapter_index
    except ImportError:
        return []
    return list(load_adapter_index(project_root).entries)


def build_model_picker_entries(state: _SessionState) -> list[ModelPickerEntry]:
    entries: list[ModelPickerEntry] = []
    for backend in managed_backend_ids():
        reachable, missing = _managed_picker_state(backend)
        _append_server_section(
            entries,
            state,
            backend=backend,
            reachable=reachable,
            missing_key=missing,
        )
    _append_server_section(
        entries,
        state,
        backend="ollama",
        reachable=ollama_reachable(),
    )
    _append_server_section(
        entries,
        state,
        backend="lmstudio",
        reachable=lmstudio_reachable(),
    )
    current_path = None
    if state.backend_id == "hf" and state.model_path.strip() and not state.model_path.startswith(
        _server_path_prefixes()
    ):
        try:
            current_path = Path(state.model_path).resolve()
        except OSError:
            current_path = None
    entries.extend(
        build_model_picker_entries_for_root(
            state.project_root,
            current_preset=state.preset_key if state.backend_id == "hf" else None,
            current_path=current_path,
            current_backend=state.backend_id,
        )
    )
    adapters = _adapter_index_entries(state.project_root)
    if adapters:
        entries.append(
            ModelPickerEntry(
                "",
                "adapters",
                str(len(adapters)),
                False,
                False,
                "header",
                "adapters",
                len(adapters),
            )
        )
        current_adapter = None
        if state.adapter_path:
            try:
                current_adapter = str(Path(state.adapter_path).resolve())
            except OSError:
                current_adapter = state.adapter_path
        for entry in adapters:
            is_current = current_adapter is not None and entry.path == current_adapter
            entries.append(
                ModelPickerEntry(
                    target=entry.name,
                    title=entry.name,
                    detail=f"{entry.preset} · {entry.dataset} · {entry.created}",
                    local=True,
                    current=is_current,
                    kind="adapter",
                    group="adapters",
                )
            )
    return entries


def _resolve_model_target(token: str, root: Path) -> tuple[str | None, Path]:
    raw = token.strip()
    if not raw:
        raise ValueError("empty model target")
    if raw.lower().startswith("hf:"):
        raw = raw[3:].strip()
    try:
        preset_key = match_preset_key(raw)
    except ValueError:
        raise
    if preset_key is not None:
        return preset_key, resolve_preset_dir(preset_key, root)
    path = Path(raw).expanduser()
    if path.exists() or any(sep in raw for sep in ("/", "\\")):
        return None, path.resolve()
    if ":" in raw and not raw.lower().startswith(_server_path_prefixes()):
        return None, path.resolve()
    raise ValueError(f"unknown preset or path: {raw!r}")


def _format_model_line(
    *,
    marker: str,
    label: str,
    repo_or_path: str,
    description: str | None = None,
) -> str:
    row = f"  {marker} {label:<28} {repo_or_path}"
    if description:
        row = f"{row}\n      {description}"
    return row


def format_model_catalog(
    state: _SessionState,
    *,
    view: str = "all",
) -> str:
    view_norm = view.strip().lower() or "all"
    known_views = ("all", "local", "downloaded", "missing", "remote", "adapters") + chat_backend_ids()
    if view_norm not in known_views:
        return "usage: /models [all | local | missing | adapters | openai | anthropic | google | ollama | lmstudio | hf]"

    lines: list[str] = [
        f"current: {_current_model_label(state)}",
        "",
    ]

    def append_server_block(backend: ChatBackendId, reachable: bool, missing_key: str | None = None) -> None:
        lines.append(f"{backend}:")
        if missing_key:
            lines.append(f"  {_env_key_set_hint(missing_key)}")
            if not is_managed_backend(backend):
                lines.append("")
                return
        if not reachable and not is_managed_backend(backend):
            lines.append("  (unreachable)")
            lines.append("")
            return
        try:
            names = list_server_models(backend)
        except Exception as exc:
            lines.append(f"  (list failed: {exc})")
            lines.append("")
            return
        if not names:
            if is_managed_backend(backend):
                lines.append("  (no local index. /models-sync)")
            else:
                lines.append("  (no models exposed)")
            lines.append("")
            return
        for name in names:
            marker = "*" if _is_current_server_entry(state, backend, name) else " "
            lines.append(_format_model_line(marker=marker, label=name, repo_or_path=f"{backend}:{name}"))
        lines.append("")

    if view_norm in managed_backend_ids():
        reachable, missing = _managed_picker_state(view_norm)
        append_server_block(view_norm, reachable, missing)
        return "\n".join(lines)
    if view_norm in ("ollama", "lmstudio"):
        append_server_block(view_norm, server_backend_reachable(view_norm))
        return "\n".join(lines)

    if view_norm == "all":
        for backend in managed_backend_ids():
            reachable, missing = _managed_picker_state(backend)
            append_server_block(backend, reachable, missing)
        append_server_block("ollama", ollama_reachable())
        append_server_block("lmstudio", lmstudio_reachable())

    root = state.project_root
    current_preset = state.preset_key if state.backend_id == "hf" else None
    current_path = None
    if state.backend_id == "hf" and state.model_path.strip():
        try:
            current_path = Path(state.model_path).resolve()
        except OSError:
            current_path = None

    downloaded: list[str] = []
    missing: list[str] = []
    for key in preset_keys_sorted():
        if preset_has_weights(key, root):
            downloaded.append(key)
        else:
            missing.append(key)

    if state.backend_id == "hf" and not chat_session_has_weights(state):
        lines.append("status: no HF weights loaded (pick a local preset or download)")
    lines.append(
        f"huggingface presets: {len(downloaded)} local, {len(missing)} not downloaded "
        f"({len(HF_MODEL_PRESETS)} registered)"
    )

    def append_preset_block(title: str, keys: list[str]) -> None:
        if not keys:
            return
        lines.append("")
        lines.append(title)
        for key in keys:
            preset = HF_MODEL_PRESETS[key]
            is_current = state.backend_id == "hf" and (
                key == current_preset
                or (
                    current_path is not None
                    and resolve_preset_dir(key, root).resolve() == current_path
                )
            )
            marker = "*" if is_current else " "
            lines.append(
                _format_model_line(
                    marker=marker,
                    label=key,
                    repo_or_path=preset.repo_id,
                    description=preset.description or None,
                )
            )

    if view_norm in ("all", "local", "downloaded"):
        append_preset_block("huggingface local:", downloaded)
    if view_norm in ("all", "missing", "remote"):
        append_preset_block("huggingface not downloaded:", missing)

    extra_dirs = local_only_model_dirs(root)
    if extra_dirs and view_norm in ("all", "local", "downloaded"):
        lines.append("")
        lines.append("local paths (no preset key):")
        for path in extra_dirs:
            marker = (
                "*"
                if state.backend_id == "hf"
                and current_path is not None
                and path.resolve() == current_path
                else " "
            )
            lines.append(_format_model_line(marker=marker, label=str(path.name), repo_or_path=str(path)))

    if view_norm in ("all", "local", "downloaded", "adapters"):
        adapters = _adapter_index_entries(root)
        if adapters:
            lines.append("")
            lines.append("adapters:")
            for entry in adapters:
                marker = (
                    "*"
                    if state.adapter_path
                    and Path(state.adapter_path).resolve() == Path(entry.path).resolve()
                    else " "
                )
                lines.append(
                    _format_model_line(
                        marker=marker,
                        label=entry.name,
                        repo_or_path=f"{entry.preset} · {entry.dataset} · {entry.created}",
                    )
                )

    return "\n".join(lines)


def _unload_model_weights(state: _SessionState) -> None:
    state.processor = None
    state.model = None
    state.tts_engine = None
    state.sst_engine = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def switch_session_model(
    state: _SessionState,
    target: str,
    *,
    on_load_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    raw = target.strip()
    parsed = _parse_server_target(raw)
    if parsed is not None:
        switch_to_server_model(state, parsed[0], parsed[1])
        return
    if raw and ":" not in raw and not raw.lower().startswith("hf:"):
        search_order: tuple[ChatBackendId, ...] = ("ollama", "lmstudio") + managed_backend_ids()
        for backend in search_order:
            if is_managed_backend(backend) and not provider_api_key(backend):
                continue
            try:
                names = list_server_models(backend)
            except Exception:
                continue
            if any(item == raw or item.lower() == raw.lower() for item in names):
                switch_to_server_model(state, backend, raw)
                return
            if any(raw.lower() in item.lower() for item in names):
                switch_to_server_model(state, backend, raw)
                return

    if state.backend_id != "hf":
        _unload_model_weights(state)
        state.backend_id = "hf"
        state.server_model = None

    try:
        preset_key, model_dir = _resolve_model_target(raw, state.project_root)
    except ValueError as exc:
        hint = _colon_provider_hint(raw)
        if hint:
            raise ValueError(f"{exc}. {hint}") from exc
        raise
    has_config = (model_dir / "config.json").is_file()
    has_weights = model_dir_has_complete_weights(model_dir) if has_config else False
    if preset_key is not None:
        if not has_config:
            _emit(
                f"preset {preset_key!r} is not on disk at {model_dir}. "
                f"Run /model-download {preset_key}"
            )
            return
        if not has_weights:
            _emit(
                f"preset {preset_key!r} is partially downloaded at {model_dir}. "
                f"Run /model-download {preset_key} to fetch the weight files."
            )
            return
    else:
        if not has_config:
            _emit(f"model directory missing config.json: {model_dir}")
            return
        if not has_weights:
            _emit(f"model directory has no complete weight files: {model_dir}")
            return

    model_path = str(model_dir.resolve())
    if model_path == state.model_path and preset_key == state.preset_key:
        _emit(f"(already using {preset_key or model_path})")
        return

    state.model_switching = True
    _unload_model_weights(state)
    try:
        _emit(f"loading model from {model_path} ...")
        processor, model = load_processor_and_model(
            model_path,
            state.quantization,
            on_load_progress=on_load_progress,
            adapter_path=state.adapter_path,
        )
        meta = read_model_meta(model_path, processor)
    except Exception as exc:
        _emit(f"(model load failed: {exc})")
        return
    finally:
        state.model_switching = False

    state.processor = processor
    state.model = model

    state.meta = meta
    state.model_path = model_path
    state.preset_key = preset_key
    state.messages = _baseline_messages(state.system_text)
    state._last_persisted_user_obj_id = 0
    state.stats = SessionStats(max_position_embeddings=meta.max_position_embeddings)
    if state.params.temperature is None:
        state.params.temperature = meta.default_temperature
    if state.params.top_p is None:
        state.params.top_p = meta.default_top_p
    if state.params.top_k is None:
        state.params.top_k = meta.default_top_k

    label = preset_key or model_path
    ctx = meta.max_position_embeddings
    ctx_note = f", context={ctx} tokens" if ctx else ""
    adapter_note = " + LoRA adapter" if state.adapter_path else ""
    _emit(f"(switched to {label}{ctx_note}{adapter_note}; conversation cleared)")


def _current_model_label(state: _SessionState) -> str:
    from cli.chat_display import model_identity_line

    extra = ""
    if is_server_backend(state.backend_id):
        extra = f"backend={state.backend_id} model={state.server_model or '(none)'}"
    elif state.preset_key:
        preset = HF_MODEL_PRESETS[state.preset_key]
        extra = f"preset={state.preset_key} repo={preset.repo_id} path={state.model_path}"
    elif state.model_path.strip():
        extra = f"path={state.model_path}"
    else:
        extra = "(no model directory; use /models or /model PATH)"
    return f"{model_identity_line(state)} | {extra}"


@_register("help", "Show this help. /help [section] filters session|model|generation|tools|speech|memory|skills|eval|train.", shortcut="?")
def _cmd_help(state: _SessionState, arg: str) -> bool:
    _ = state
    _emit(_format_help(arg))
    return False


@_register("backend", "Show or switch chat backend: /backend [auto|ollama|lmstudio|hf|openai|anthropic|google].")
def _cmd_backend(state: _SessionState, arg: str) -> bool:
    token = arg.strip()
    if not token:
        _emit(f"backend={state.backend_id}  (set SOPHON_CHAT_BACKEND={backend_help_tokens()})")
        return False
    if token.lower() == "auto":
        try:
            resolved = resolve_chat_backend("auto")
        except ValueError as exc:
            _emit(str(exc))
            return False
        switch_session_backend(state, resolved)
        _emit(f"(auto resolved to {resolved})")
        return False
    switch_session_backend(state, token)
    return False


@_register("models", "List all sources: openai, anthropic, google, ollama, lmstudio, huggingface.")
def _cmd_models(state: _SessionState, arg: str) -> bool:
    _emit(format_model_catalog(state, view=arg.strip() or "all"))
    return False


@_register(
    "models-sync",
    "Refresh the versioned OpenAI/Anthropic/Google model index from the APIs. /models-sync [openai|anthropic|google].",
    aliases=("models-refresh",),
)
def _cmd_models_sync(state: _SessionState, arg: str) -> bool:
    _ = state
    from backend.model_index import managed_index_path

    token = arg.strip().lower()
    backends: tuple[str, ...] | None = None
    if token:
        if token not in managed_backend_ids():
            _emit("usage: /models-sync [openai | anthropic | google]")
            return False
        backends = (token,)
    _emit("syncing managed model index (API list) ...")
    result = sync_managed_model_index(backends)
    wanted = backends or managed_backend_ids()
    for backend in wanted:
        names = result.get(backend, [])
        if not provider_api_key(backend):
            _emit(f"  {backend}: skipped ({provider_key_hint(backend)} missing)")
            continue
        _emit(f"  {backend}: {len(names)} chat models")
    _emit(f"wrote {managed_index_path()}")
    return False


@_register("model", "Show or switch model: /model [openai:NAME | anthropic:NAME | google:NAME | ollama:NAME | lmstudio:NAME | PRESET | PATH].")
def _cmd_model(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(_current_model_label(state))
        return False
    try:
        switch_session_model(state, arg)
    except ValueError as exc:
        _emit(str(exc))
    return False


@_register(
    "setup",
    "Device recommender: /setup reads and prints top 10, then install rank 1 or skip. /setup apply | skip | why.",
)
def _cmd_setup(state: _SessionState, arg: str) -> bool:
    from cli.setup_commands import handle_setup

    return handle_setup(state, arg)


@_register("model-download", "Download a preset from Hugging Face Hub: /model-download PRESET.")
def _cmd_model_download(state: _SessionState, arg: str) -> bool:
    token = arg.strip()
    if not token:
        _emit("usage: /model-download <preset>")
        return False
    try:
        preset_key = match_preset_key(token)
    except ValueError as exc:
        _emit(str(exc))
        return False
    if preset_key is None:
        _emit(f"unknown preset: {token!r}")
        return False
    model_dir = resolve_preset_dir(preset_key, state.project_root)
    if preset_has_weights(preset_key, state.project_root):
        _emit(f"(preset {preset_key!r} already on disk)")
        return False
    preset = HF_MODEL_PRESETS[preset_key]
    if (model_dir / "config.json").is_file():
        _emit(f"resuming download for partially fetched preset {preset_key} ({preset.repo_id}) ...")
    else:
        _emit(f"downloading {preset_key} ({preset.repo_id}) ...")
    state.model_switching = True
    try:
        configure_hub_verbose(force_tqdm=True)
        os.environ["TQDM_POSITION"] = "-1"

        def on_download_status(text: str) -> None:
            _emit(f"hub: {text}")

        progress_total = {"value": 0}
        progress_n = {"value": 0}
        rate_state = {"t": time.monotonic(), "n": 0, "bps": 0.0, "grew_at": time.monotonic()}

        def emit_progress(n: int, total: int, phase: str) -> None:
            if n > progress_n["value"]:
                progress_n["value"] = n
            shown_n = progress_n["value"]
            shown_total = total if total >= 4097 else progress_total["value"]
            if shown_total > progress_total["value"]:
                progress_total["value"] = shown_total
            shown_total = progress_total["value"]
            now = time.monotonic()
            dt = now - rate_state["t"]
            dn = shown_n - rate_state["n"]
            if dn > 0:
                rate_state["grew_at"] = now
                if dt >= 0.4:
                    inst = dn / dt
                    prev = rate_state["bps"]
                    rate_state["bps"] = inst if prev <= 0 else (0.65 * prev + 0.35 * inst)
                    rate_state["t"] = now
                    rate_state["n"] = shown_n
            elif now - rate_state["grew_at"] > 8.0:
                rate_state["bps"] = 0.0
            bps = float(rate_state["bps"])
            status = format_download_progress(shown_n, shown_total, phase, bytes_per_s=bps)
            sys.stderr.write(f"\r[sophon] {status.replace(chr(10), ' | ')}    ")
            sys.stderr.flush()

        def on_expected_bytes(nbytes: int) -> None:
            if nbytes > progress_total["value"]:
                progress_total["value"] = nbytes
            emit_progress(
                scan_local_dir_download_bytes(model_dir),
                progress_total["value"],
                "Hub size known",
            )

        def on_download_progress(n: int, total: int, label: str) -> None:
            phase, _ = split_hub_progress_label(label)
            if is_file_count_progress(n, total, phase) or is_file_count_progress(n, total, label):
                status = format_download_progress(n, total, phase)
                sys.stderr.write(f"\r[sophon] {status.replace(chr(10), ' | ')}    ")
                sys.stderr.flush()
                return
            if total >= 4097 and total > progress_total["value"]:
                progress_total["value"] = total
            emit_progress(n, progress_total["value"], phase)

        def on_disk_bytes(nbytes: int) -> None:
            emit_progress(nbytes, progress_total["value"], "on-disk download")

        def on_hub_log(text: str) -> None:
            hinted = parse_hub_size_hint(text)
            if hinted > progress_total["value"]:
                progress_total["value"] = hinted
                emit_progress(progress_n["value"], hinted, "Hub size hint")
            _emit(text)

        tqdm_class = hub_tqdm_bridge_factory(throttled_progress_callback(on_download_progress))
        with watch_local_download_bytes(model_dir, on_disk_bytes):
            with capture_hub_download_logs(on_hub_log):
                with capture_hub_user_warnings(lambda text: _emit(f"(warning: {text})")):
                    path = download_preset_snapshot(
                        preset_key,
                        tqdm_class=tqdm_class,
                        verbose=True,
                        on_status=on_download_status,
                        on_expected_bytes=on_expected_bytes,
                    )
        sys.stderr.write("\n")
        sys.stderr.flush()
    except Exception as exc:
        sys.stderr.write("\n")
        sys.stderr.flush()
        _emit(f"(download failed: {exc})")
        return False
    finally:
        state.model_switching = False
    _emit(f"(downloaded to {path}; use /model {preset_key} to load)")
    return False


def _parse_finetune_command(
    arg: str,
) -> tuple[str | None, str | None, str | None, list[str]]:
    tokens = arg.split()
    dataset_id: str | None = None
    preset_key: str | None = None
    recipe_path: str | None = None
    overrides: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        lower = token.lower()
        if lower == "on":
            idx += 1
            if idx >= len(tokens):
                raise ValueError("usage: /finetune [DATASET] on PRESET [from PATH] [KEY=VAL ...]")
            try:
                matched_preset = match_preset_key(tokens[idx])
            except ValueError:
                matched_preset = None
            preset_key = matched_preset or tokens[idx]
            idx += 1
            continue
        if lower == "from":
            idx += 1
            if idx >= len(tokens):
                raise ValueError("usage: /finetune from PATH")
            recipe_path = tokens[idx]
            idx += 1
            continue
        if "=" in token:
            overrides.append(token)
            idx += 1
            continue
        from training.finetune.datasets.registry import match_dataset_preset

        matched = match_dataset_preset(token)
        if matched is not None and dataset_id is None:
            dataset_id = matched
            idx += 1
            continue
        raise ValueError(f"unrecognized finetune token {token!r}")
    return dataset_id, preset_key, recipe_path, overrides


def _format_finetune_dataset_catalog() -> str:
    from training.finetune.datasets.load import dataset_source_summary
    from training.finetune.datasets.registry import FINETUNE_DATASET_PRESETS, dataset_preset_keys_sorted
    from utils.device.env_bootstrap import sophon_project_root

    root = sophon_project_root()
    lines = ["finetune datasets:"]
    for key in dataset_preset_keys_sorted():
        preset = FINETUNE_DATASET_PRESETS[key]
        lines.append(f"  {key}: {dataset_source_summary(preset, root)} max_examples={preset.max_examples}")
        lines.append(f"    {preset.description}")
    lines.append("")
    lines.append("usage: /finetune [DATASET] [on PRESET] [from PATH] [KEY=VAL ...]")
    lines.append("  default DATASET=gsm8k_instructions unless `from PATH` supplies stages")
    lines.append("  overrides: max_examples=500 epochs=1 lr=2e-4 batch_size=2 continue_from=NAME")
    return "\n".join(lines)


def _record_train_progress(state: _SessionState, step: int, total: int, label: str) -> None:
    if "loss=" in label:
        try:
            raw = label.split("loss=", 1)[1].split()[0]
            state.train_loss_history.append(float(raw))
        except (IndexError, ValueError):
            pass
    if total > 0:
        _emit(f"train: {label} ({step}/{total})")
    else:
        _emit(f"train: {label}")


def _reload_current_session_model(
    state: _SessionState,
    *,
    on_load_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    if state.preset_key:
        switch_session_model(state, state.preset_key, on_load_progress=on_load_progress)
        return
    if state.model_path.strip():
        switch_session_model(state, state.model_path, on_load_progress=on_load_progress)
        return
    _emit("(no model selected; use /model PRESET first)")


@_register(
    "trainer",
    "Trainer mode: /trainer | use | select | data get KEY | run [full] | watch | stop.",
)
def _cmd_trainer(state: _SessionState, arg: str) -> bool:
    from cli.trainer_commands import handle_trainer

    return handle_trainer(state, arg)


@_register("finetune-datasets", "List finetune dataset presets.")
def _cmd_finetune_datasets(state: _SessionState, _arg: str) -> bool:
    _ = state
    _emit(_format_finetune_dataset_catalog())
    return False


@_register(
    "finetune",
    "LoRA finetune: /finetune [DATASET] [on PRESET] [from PATH] [KEY=VAL ...].",
)
def _cmd_finetune(state: _SessionState, arg: str) -> bool:
    try:
        dataset_id, preset_key, recipe_path, overrides = _parse_finetune_command(arg)
    except ValueError as exc:
        _emit(str(exc))
        return False
    if preset_key is None and not recipe_path:
        preset_key = state.preset_key
    if not preset_key and not recipe_path:
        _emit("usage: /finetune [DATASET] on PRESET [from PATH] [KEY=VAL ...]  (or select a preset first)")
        return False

    from training.finetune.job import run_finetune_job

    _unload_model_weights(state)
    state.finetune_running = True
    state.train_loss_history = []
    try:
        _emit(
            f"starting finetune preset={preset_key!r} dataset={dataset_id!r} "
            f"from={recipe_path or '-'}"
        )

        def on_log(msg: str) -> None:
            _emit(msg)

        result = run_finetune_job(
            preset_key,
            dataset_id,
            project_root=state.project_root,
            backend_id="auto",
            recipe_path=recipe_path,
            recipe_override_tokens=overrides,
            on_progress=lambda step, total, label: _record_train_progress(state, step, total, label),
            on_log=on_log,
        )
    except Exception as exc:
        _emit(f"(finetune failed: {exc})")
        return False
    finally:
        state.finetune_running = False

    state.preset_key = result.meta.preset_key
    state.last_finetune_run_dir = result.run_dir
    state.last_finetune_adapter_name = result.adapter_name
    state.adapter_path = str(result.adapter_dir.resolve())
    _emit(f"(finetune done; adapter at {result.adapter_dir})")
    load_name = result.adapter_name or str(result.adapter_dir)
    _emit(f"(backend={result.backend_id}; /adapter load {load_name})")
    return False


@_register("finetune-config", "Print the resolved finetune recipe. /finetune-config [from PATH] [KEY=VAL ...].")
def _cmd_finetune_config(state: _SessionState, arg: str) -> bool:
    try:
        dataset_id, preset_key, recipe_path, overrides = _parse_finetune_command(arg)
    except ValueError as exc:
        _emit(str(exc))
        return False
    if preset_key is None and not recipe_path:
        preset_key = state.preset_key
    from training.finetune.recipe import recipe_to_mapping, resolve_finetune_recipe
    import yaml

    recipe = resolve_finetune_recipe(
        project_root=state.project_root,
        dataset_id=dataset_id,
        preset_key=preset_key,
        recipe_path=recipe_path,
        override_tokens=overrides,
    )
    path = recipe.recipe_path or str(state.project_root / "config" / "finetune" / "default.yaml")
    _emit(f"recipe file: {path}")
    _emit(yaml.safe_dump(recipe_to_mapping(recipe), sort_keys=False).rstrip())
    return False


@_register("finetune-status", "Show last finetune run summary. /finetune-status [NAME] [curves].")
def _cmd_finetune_status(state: _SessionState, arg: str) -> bool:
    from training.common.curves import losses_from_metrics_jsonl, sparkline
    from training.common.index import resolve_adapter_entry
    from training.common.types import RunMeta

    parts = arg.split()
    want_curves = any(p.lower() == "curves" for p in parts)
    name_parts = [p for p in parts if p.lower() != "curves"]
    run_dir = state.last_finetune_run_dir
    adapter_name = state.last_finetune_adapter_name
    if name_parts:
        entry = resolve_adapter_entry(name_parts[0], project_root=state.project_root)
        if entry is None:
            _emit(f"(adapter not found: {name_parts[0]})")
            return False
        run_dir = Path(entry.path)
        adapter_name = entry.name
    if run_dir is None:
        _emit("(no finetune run in this session; pass NAME or /adapter list)")
        return False
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        _emit(f"(run dir exists but no run_meta.json: {run_dir})")
        return False
    meta = RunMeta.read_json(meta_path)
    _emit(f"run_dir: {run_dir}")
    if adapter_name or meta.adapter_name:
        _emit(f"name: {adapter_name or meta.adapter_name}")
    _emit(f"preset: {meta.preset_key}  dataset: {meta.dataset_id}  backend: {meta.backend_id}")
    if meta.stage_id:
        _emit(f"stage: {meta.stage_id}  parent: {meta.parent_adapter or '-'}")
    if meta.train_metrics:
        _emit(f"train_metrics: {meta.train_metrics}")
    if state.adapter_path:
        _emit(f"adapter_path: {state.adapter_path}")
    losses = losses_from_metrics_jsonl(run_dir / "metrics.jsonl")
    if losses:
        _emit(f"last_loss: {losses[-1]:.4f}  steps_logged: {len(losses)}")
        _emit(f"curve: {sparkline(losses)}")
    elif want_curves:
        _emit("(no metrics.jsonl losses to plot)")
    return False


@_register("adapter", "LoRA adapter: /adapter show | list | load [NAME|PATH] | clear.")
def _cmd_adapter(state: _SessionState, arg: str) -> bool:
    tokens = arg.strip().split(maxsplit=1)
    action = tokens[0].lower() if tokens else "show"
    rest = tokens[1].strip() if len(tokens) > 1 else ""

    if action in ("", "show"):
        if state.adapter_path:
            _emit(f"adapter_path = {state.adapter_path}")
        else:
            _emit("adapter_path = (none)")
        if state.last_finetune_run_dir is not None:
            _emit(f"last_finetune_run_dir = {state.last_finetune_run_dir}")
        if state.last_finetune_adapter_name:
            _emit(f"last_adapter_name = {state.last_finetune_adapter_name}")
        return False

    if action == "list":
        from training.common.index import load_adapter_index

        entries = load_adapter_index(state.project_root).entries
        if not entries:
            _emit("no adapters in data/training/adapters/index.json")
            return False
        for entry in entries:
            loss = f" loss={entry.last_loss:.4f}" if entry.last_loss is not None else ""
            _emit(f"  {entry.name}: {entry.preset} / {entry.dataset} {entry.created}{loss}")
            _emit(f"    {entry.path}")
        return False

    if action == "clear":
        state.adapter_path = None
        if state.preset_key or state.model_path:
            _reload_current_session_model(state)
        else:
            _emit("(adapter cleared)")
        return False

    if action == "load":
        from training.common.index import resolve_adapter_entry

        target_preset = state.preset_key
        if rest:
            entry = resolve_adapter_entry(rest, project_root=state.project_root)
            if entry is None:
                _emit(f"adapter not found: {rest}")
                return False
            state.adapter_path = entry.path
            if entry.preset:
                target_preset = entry.preset
        elif state.adapter_path:
            pass
        elif state.last_finetune_run_dir is not None:
            state.adapter_path = str(state.last_finetune_run_dir.resolve())
        else:
            _emit("usage: /adapter load [NAME|PATH]  (or run /finetune first)")
            return False
        if not target_preset and not state.model_path.strip():
            _emit(f"(adapter set to {state.adapter_path}; use /model PRESET to load)")
            return False
        if target_preset and target_preset != state.preset_key:
            switch_session_model(state, target_preset)
            return False
        _reload_current_session_model(state)
        return False

    _emit("usage: /adapter show | list | load [NAME|PATH] | clear")
    return False


@_register("quit", "Exit chat.", shortcut="q")
def _cmd_quit(state: _SessionState, _arg: str) -> bool:
    state.exit_requested = True
    return False


@_register("reset", "Clear conversation (keeps system prompt; memory DB untouched).")
def _cmd_reset(state: _SessionState, _arg: str) -> bool:
    state.messages = _baseline_messages(state.system_text)
    state._last_persisted_user_obj_id = 0
    _wipe_transcript_ui()
    _emit("(conversation cleared)")
    return False


@_register("log", "Show session log path and tail. /log path | /log [N].", aliases=("logs",))
def _cmd_log(state: _SessionState, arg: str) -> bool:
    _ = state
    from cli.host.session_log import resolve_session_log_path
    from utils.device.platform import to_linux_display_path

    path = resolve_session_log_path()
    display = to_linux_display_path(path)
    token = arg.strip()
    parts = token.split()
    if token.lower() in ("path", "where"):
        _emit(display)
        return False
    n = 80
    rest = token
    if parts and parts[0].lower() in ("tail", "head"):
        rest = " ".join(parts[1:])
    if rest.strip():
        try:
            n = int(rest.strip())
        except ValueError:
            _emit("usage: /log [path | N | tail N]")
            return False
    n = max(1, min(n, 200))
    _emit(f"log {display}")
    if not path.is_file():
        _emit("(log file not found)")
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    tail = lines[-n:]
    if not tail:
        _emit("(empty)")
        return False
    _emit("\n".join(tail))
    return False


@_register("clear", "Empty the visible chat and start a new transcript. Durable memory stays.")
def _cmd_clear(state: _SessionState, _arg: str) -> bool:
    new_id = _dt.datetime.now().strftime("session_%Y%m%d_%H%M%S")
    if state.memory_layer is not None:
        scope = state.memory_layer.rotate_session(new_id)
        state.memory_scope = scope
    elif state.memory_scope is not None:
        state.memory_scope = MemoryScope(session_id=new_id, user_id=state.memory_scope.user_id)
    state.messages = _baseline_messages(state.system_text)
    state._last_persisted_user_obj_id = 0
    _wipe_transcript_ui()
    _emit(f"(chat cleared, session={new_id})")
    return False


@_register("save", "Save transcript to <path> or data/chat_logs/<timestamp>.txt", shortcut="s")
def _cmd_save(state: _SessionState, arg: str) -> bool:
    target = arg.strip()
    if not target:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        target = str(sophon_chat_logs_dir() / f"{ts}.txt")
    p = Path(target).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for m in state.messages:
        role = str(m.get("role", "?"))
        content = m.get("content", "")
        lines.append(f"=== {role} ===")
        lines.append(str(content))
        lines.append("")
    try:
        p.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        _emit(f"save failed: {exc}")
        return False
    _emit(f"saved transcript -> {p}")
    return False


@_register("pop", "Drop the last user+assistant turn.", shortcut="u")
def _cmd_pop(state: _SessionState, _arg: str) -> bool:
    dropped = 0
    while state.messages and dropped < 2:
        role = state.messages[-1].get("role")
        state.messages.pop()
        dropped += 1
        if role == "user":
            break
    _emit(f"(popped {dropped} message(s))")
    return False


@_register("regen", "Pop last assistant reply and regenerate.", shortcut="r")
def _cmd_regen(state: _SessionState, _arg: str) -> bool:
    while state.messages and state.messages[-1].get("role") != "user":
        state.messages.pop()
    if not state.messages or state.messages[-1].get("role") != "user":
        _emit("(nothing to regenerate)")
        return False
    return True


@_register("debug", "Toggle debug stats footer. Use /debug on or /debug off.", shortcut="d")
def _cmd_debug(state: _SessionState, arg: str) -> bool:
    a = arg.strip().lower()
    if a == "on":
        state.debug = True
    elif a == "off":
        state.debug = False
    else:
        state.debug = not state.debug
    _emit(f"(debug = {'on' if state.debug else 'off'})")
    return False


@_register("stats", "Print verbose stats. /stats reset zeroes counters.")
def _cmd_stats(state: _SessionState, arg: str) -> bool:
    if arg.strip().lower() == "reset":
        state.stats.reset()
        _emit("(stats reset)")
        return False
    _emit(state.stats.format_table())
    return False


@_register("tokens", "Print current context tokens vs window.")
def _cmd_tokens(state: _SessionState, _arg: str) -> bool:
    cur = state.stats.last_ctx_tokens
    mx = state.stats.max_position_embeddings
    if mx:
        pct = 100.0 * cur / mx if cur else 0.0
        _emit(f"context: {cur} / {mx} tokens ({pct:.1f}%)")
    else:
        _emit(f"context: {cur} tokens (window unknown)")
    return False


@_register("system", "Set/show/clear system prompt. /system show | /system clear | /system <text>")
def _cmd_system(state: _SessionState, arg: str) -> bool:
    a = arg.strip()
    if a == "" or a.lower() == "show":
        _emit(f"system: {state.system_text!r}")
        return False
    if a.lower() == "clear":
        state.system_text = None
    else:
        state.system_text = a
    state.messages = _baseline_messages(state.system_text)
    state._last_persisted_user_obj_id = 0
    _emit("(system updated, conversation reset)")
    return False


@_register("rag-status", "Show retrieval backend + last query metrics.")
def _cmd_rag_status(state: _SessionState, _arg: str) -> bool:
    if state.retriever is None and state.structure_retriever is None:
        from processing.text.retrieval.corpus import default_index_exists, default_leann_index_path

        default_path = default_leann_index_path()
        if default_index_exists(default_path):
            _emit(
                f"retrieval: disabled in this session, but default index exists at {default_path}. "
                "Restart chat (or /rag-index) to bind it, or pass --rag leann."
            )
        else:
            _emit(
                "retrieval: disabled (no index). "
                "Set SOPHON_VAULT_PATH then run /rag-index --rebuild "
                f"(writes {default_path}). Obsidian tools are separate from RAG."
            )
        return False
    primary = state.retriever.backend_id() if state.retriever is not None else "none"
    structure = (
        state.structure_retriever.backend_id() if state.structure_retriever is not None else "none"
    )
    _emit(
        f"retrieval: backend={primary} structure={structure} "
        f"adaptive={'on' if state.rag_adaptive else 'off'} top_k={state.retrieval_top_k}"
    )
    decision = state.last_retrieval_decision
    if isinstance(decision, dict) and decision:
        _emit(
            f"last decision: {decision.get('decision')} "
            f"(reason={decision.get('reason')})"
        )
    else:
        _emit("last decision: (none yet)")
    last = state.last_retrieval_metrics
    if isinstance(last, dict) and last:
        from processing.text.retrieval.metrics import RetrievalQueryMetrics, format_query_metrics

        try:
            metrics = RetrievalQueryMetrics(
                latency_s=float(last.get("latency_s") or 0.0),
                n_hits=int(last.get("n_hits") or 0),
                top_k=int(last.get("top_k") or state.retrieval_top_k),
                query_chars=int(last.get("query_chars") or 0),
                top_score=last.get("top_score"),
                mean_score=last.get("mean_score"),
                min_score=last.get("min_score"),
                score_margin=last.get("score_margin"),
                backend_id=str(last.get("backend_id") or ""),
                extras=dict(last.get("extras") or {}),
            )
            _emit(f"last query: {format_query_metrics(metrics)}")
        except Exception:
            _emit(f"last query: {last}")
    else:
        _emit("last query: (none yet — send a chat turn or /rag-probe)")
    hist = state.retrieval_probe_history
    if len(hist) >= 2:
        from processing.text.retrieval.metrics import RetrievalQueryMetrics, summarize_probe

        rows = []
        for item in hist:
            if not isinstance(item, dict):
                continue
            try:
                rows.append(
                    RetrievalQueryMetrics(
                        latency_s=float(item.get("latency_s") or 0.0),
                        n_hits=int(item.get("n_hits") or 0),
                        top_k=int(item.get("top_k") or state.retrieval_top_k),
                        query_chars=int(item.get("query_chars") or 0),
                        top_score=item.get("top_score"),
                        mean_score=item.get("mean_score"),
                        min_score=item.get("min_score"),
                        score_margin=item.get("score_margin"),
                        backend_id=str(item.get("backend_id") or ""),
                    )
                )
            except Exception:
                continue
        if rows:
            summary = summarize_probe(rows)
            _emit("session probe window:")
            for line in summary.format_lines():
                _emit(f"  {line}")
    return False


@_register(
    "rag-probe",
    "Estimate retrieval query cost: /rag-probe [N] [query text]. Uses last user turn if query omitted.",
)
def _cmd_rag_probe(state: _SessionState, arg: str) -> bool:
    active = state.retriever or state.structure_retriever
    if active is None:
        _emit("retrieval disabled — set --rag leann --rag-index PATH first")
        return False
    from processing.text.retrieval.metrics import (
        RetrievalQueryMetrics,
        build_query_metrics,
        summarize_probe,
    )

    parts = arg.split()
    repeats = 1
    query = ""
    if parts and parts[0].isdigit():
        repeats = max(int(parts[0]), 1)
        query = arg[len(parts[0]) :].strip()
    else:
        query = arg.strip()
    if not query:
        query = _latest_user_text(state) or ""
    if not query.strip():
        _emit("usage: /rag-probe [N] <query>   (or run after a user turn)")
        return False
    repeats = min(repeats, 50)
    _emit(
        f"(rag-probe backend={active.backend_id()} n={repeats} "
        f"top_k={state.retrieval_top_k} query_chars={len(query)})"
    )
    rows: list[RetrievalQueryMetrics] = []
    for i in range(repeats):
        try:
            import time

            t0 = time.perf_counter()
            result = active.retrieve(
                RetrievalQuery(text=query, params={"top_k": state.retrieval_top_k}),
            )
            latency = time.perf_counter() - t0
            raw = result.extras.get("metrics") if isinstance(result.extras, dict) else None
            if isinstance(raw, dict):
                metrics = RetrievalQueryMetrics(
                    latency_s=float(raw.get("latency_s") or latency),
                    n_hits=int(raw.get("n_hits") or len(result.chunks)),
                    top_k=int(raw.get("top_k") or state.retrieval_top_k),
                    query_chars=int(raw.get("query_chars") or len(query)),
                    top_score=raw.get("top_score"),
                    mean_score=raw.get("mean_score"),
                    min_score=raw.get("min_score"),
                    score_margin=raw.get("score_margin"),
                    backend_id=str(raw.get("backend_id") or active.backend_id()),
                    extras=dict(raw.get("extras") or {}),
                )
            else:
                metrics = build_query_metrics(
                    latency_s=latency,
                    chunks=list(result.chunks),
                    top_k=state.retrieval_top_k,
                    query_chars=len(query),
                    backend_id=active.backend_id(),
                )
                result.extras["metrics"] = metrics.as_dict()
            rows.append(metrics)
            state.last_retrieval_metrics = metrics.as_dict()
            state.retrieval_probe_history.append(metrics.as_dict())
            if repeats <= 5:
                _emit(f"  [{i + 1}] latency={metrics.latency_s:.3f}s hits={metrics.n_hits} top={metrics.top_score}")
        except Exception as exc:
            _emit(f"  [{i + 1}] failed: {exc}")
    if len(state.retrieval_probe_history) > 64:
        state.retrieval_probe_history = state.retrieval_probe_history[-64:]
    if rows:
        summary = summarize_probe(rows)
        _emit("summary:")
        for line in summary.format_lines():
            _emit(f"  {line}")
        _emit(
            "note: latency/score/margin estimate query cost and confidence. "
            "Labeled hit@k needs a JSONL of {query, relevant_ids}."
        )
    return False


@_register("rag-index", "Build default vault+project LEANN corpus: /rag-index [--rebuild].")
def _cmd_rag_index(state: _SessionState, arg: str) -> bool:
    from processing.text.retrieval.corpus import build_default_corpus

    tokens = arg.split()
    rebuild = any(t in ("--rebuild", "rebuild") for t in tokens)
    _emit(f"(rag-index starting rebuild={rebuild})...")
    try:
        result = build_default_corpus(
            rebuild=rebuild,
            on_progress=lambda msg: _emit(f"[rag-index] {msg}"),
        )
    except Exception as exc:
        _emit(f"(rag-index failed: {exc})")
        return False
    _emit(
        f"(rag-index done index={result.index_path} files={result.n_files} "
        f"chunks={result.n_chunks})"
    )
    for line in result.messages:
        _emit(line)
    if state.retriever is None:
        try:
            from processing.text.retrieval import load_rag_retriever

            candidate = load_rag_retriever("leann", native_index_path=result.index_path)
            if candidate.backend_id() != "noop":
                state.retriever = candidate
                _emit(f"(retrieval enabled backend={candidate.backend_id()})")
        except Exception as exc:
            _emit(f"(retrieval re-init failed: {exc})")
    return False


@_register("eval", "Benchmarks: /eval | /eval model [task] [N] | /eval rag [N] | /eval train.")
def _cmd_eval(state: _SessionState, arg: str) -> bool:
    from cli.eval_commands import (
        available_model_tasks,
        format_eval_help,
        format_train_index_lines,
        run_model_eval,
        run_rag_eval,
        summarize_task_result,
    )

    parts = arg.split()
    if not parts:
        _emit(format_eval_help())
        tasks = available_model_tasks()
        if tasks:
            _emit(f"available tasks: {', '.join(tasks)}")
        if state.last_eval_summary:
            _emit(f"last eval: {state.last_eval_summary}")
        return False

    suite = parts[0].lower()
    if suite == "train":
        for line in format_train_index_lines():
            _emit(line)
        _emit("use /finetune to start training; /adapter load NAME to attach a LoRA")
        return False

    if state.eval_running or state.finetune_running:
        _emit("(eval/finetune already running)")
        return False

    from backend.chat_resolve import is_server_backend

    if is_server_backend(state.backend_id):
        backend = state.backend_id
        server_model = state.server_model
        model_path = None
        preset_key = None
    else:
        backend = "hf"
        server_model = None
        model_path = state.model_path or None
        preset_key = state.preset_key
    quantization = state.quantization or "none"

    if suite == "model":
        task_id = "hellaswag"
        limit: int | None = None
        if len(parts) >= 2 and not parts[1].isdigit():
            task_id = parts[1]
            if len(parts) >= 3 and parts[2].isdigit():
                limit = int(parts[2])
        elif len(parts) >= 2 and parts[1].isdigit():
            limit = int(parts[1])
        state.eval_running = True
        try:
            result = run_model_eval(
                task_id=task_id,
                limit=limit,
                backend=backend,
                model_path=model_path,
                preset_key=preset_key,
                quantization=quantization,
                server_model=server_model,
                on_emit=_emit,
            )
            state.last_eval_summary = summarize_task_result(result)
            _emit(
                f"(eval model done task={result.task_id} acc={result.accuracy:.3f} "
                f"n={result.total} out={result.summary_path})"
            )
        except Exception as exc:
            _emit(f"(eval model failed: {exc})")
        finally:
            state.eval_running = False
        return False

    if suite == "rag":
        limit = 20
        if len(parts) >= 2 and parts[1].isdigit():
            limit = int(parts[1])
        state.eval_running = True
        try:
            result = run_rag_eval(
                limit=limit,
                backend=backend,
                model_path=model_path,
                preset_key=preset_key,
                quantization=quantization,
                server_model=server_model,
                on_emit=_emit,
            )
            state.last_eval_summary = summarize_task_result(result)
            extra = ""
            try:
                import json

                summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
                extra = (
                    f" acc_off={float(summary.get('accuracy_no_rag') or 0):.3f} "
                    f"acc_on={float(summary.get('accuracy_with_rag') or 0):.3f} "
                    f"hit_rate={float(summary.get('retrieval_hit_rate') or 0):.3f}"
                )
            except Exception:
                pass
            _emit(f"(eval rag done n={result.total}{extra} out={result.summary_path})")
        except Exception as exc:
            _emit(f"(eval rag failed: {exc})")
        finally:
            state.eval_running = False
        return False

    _emit(f"unknown eval suite {suite!r}")
    _emit(format_eval_help())
    return False


@_register("obsidian-status", "Show Obsidian Local REST API tool status.")
def _cmd_obsidian_status(state: _SessionState, _arg: str) -> bool:
    _ = state
    from integrations.obsidian.client import (
        ObsidianClient,
        obsidian_api_key,
        obsidian_api_url,
        obsidian_tools_enabled,
    )

    enabled = obsidian_tools_enabled()
    url = obsidian_api_url()
    key = obsidian_api_key()
    _emit(f"obsidian tools: {'on' if enabled else 'off'} (SOPHON_OBSIDIAN_TOOLS)")
    _emit(f"api url: {url}")
    _emit(f"api key: {'set' if key else 'missing'}")
    if not enabled:
        _emit("enable with SOPHON_OBSIDIAN_TOOLS=1 and restart chat")
        return False
    try:
        info = ObsidianClient(timeout_s=5.0).ping()
        _emit(f"reachable: yes ({info})")
    except Exception as exc:
        _emit(f"reachable: no ({exc})")
    return False


@_register("zotero-status", "Show Zotero local API / sqlite tool status.")
def _cmd_zotero_status(state: _SessionState, _arg: str) -> bool:
    _ = state
    from integrations.zotero.client import (
        ZoteroClient,
        zotero_api_url,
        zotero_db_path,
        zotero_tools_enabled,
    )

    enabled = zotero_tools_enabled()
    ping = ZoteroClient().ping()
    _emit(f"zotero tools: {'on' if enabled else 'off'} (SOPHON_ZOTERO_TOOLS, default on if library is reachable)")
    _emit(f"api url: {zotero_api_url()}")
    db = zotero_db_path()
    _emit(f"sqlite: {db if db is not None else 'missing'}")
    _emit(f"backend: {ping.get('backend')} ok={ping.get('ok')}")
    if ping.get("error"):
        _emit(f"error: {ping.get('error')}")
    if not enabled:
        _emit("set SOPHON_ZOTERO_TOOLS=1 or keep the library reachable. SOPHON_ZOTERO_TOOLS=0 disables.")
    _emit("slash: /zotero-tree /zotero-search QUERY /zotero-metrics /zotero-read KEY")
    return False


@_register("zotero-tree", "Print Zotero collection folders. /zotero-tree [collection] [depth].")
def _cmd_zotero_tree(state: _SessionState, arg: str) -> bool:
    _ = state
    from integrations.zotero.client import ZoteroClient, format_tool_payload

    parts = [bit for bit in arg.split() if bit]
    depth = 2
    collection = ""
    if parts and parts[-1].isdigit():
        depth = int(parts[-1])
        parts = parts[:-1]
    if parts:
        collection = " ".join(parts)
    payload = ZoteroClient().format_tree(collection=collection, depth=depth)
    _emit(format_tool_payload(payload))
    return False


@_register("zotero-search", "Search Zotero titles, authors, keys, and indexed PDFs. /zotero-search QUERY")
def _cmd_zotero_search(state: _SessionState, arg: str) -> bool:
    _ = state
    from integrations.zotero.client import ZoteroClient, format_tool_payload

    query = arg.strip()
    if not query:
        _emit("usage: /zotero-search QUERY")
        return False
    payload = ZoteroClient().search_items(query, limit=20)
    _emit(format_tool_payload(payload))
    return False


@_register("zotero-metrics", "Local Zotero citation coverage: counts, types, years, DOIs, citekeys.")
def _cmd_zotero_metrics(state: _SessionState, _arg: str) -> bool:
    _ = state
    from integrations.zotero.client import ZoteroClient, format_tool_payload

    _emit(format_tool_payload(ZoteroClient().library_metrics()))
    return False


@_register("zotero-read", "Read one Zotero item. /zotero-read KEY|TITLE [--pdf]")
def _cmd_zotero_read(state: _SessionState, arg: str) -> bool:
    _ = state
    from integrations.zotero.client import ZoteroClient

    raw = arg.strip()
    include_pdf = False
    if raw.endswith("--pdf"):
        include_pdf = True
        raw = raw[: -len("--pdf")].strip()
    if not raw:
        _emit("usage: /zotero-read KEY|TITLE [--pdf]")
        return False
    _emit(ZoteroClient().item_record(raw, include_pdf_text=include_pdf))
    return False


@_register("google-status", "Show Google OAuth / bookmarks / SearXNG / Custom Search status.")
def _cmd_google_status(state: _SessionState, _arg: str) -> bool:
    _ = state
    from integrations.google.bookmarks import bookmarks_available, bookmarks_path
    from integrations.google.oauth import format_accounts_status, google_tools_enabled
    from integrations.google.paths import client_secrets_path
    from integrations.google.search import cse_configured, web_search_tools_enabled
    from integrations.google.searxng import searxng_base_url, searxng_configured

    status = format_accounts_status()
    _emit(f"google tools: {'on' if google_tools_enabled() else 'off'} (SOPHON_GOOGLE_TOOLS)")
    _emit(f"client secrets: {client_secrets_path() or 'missing'}")
    _emit(f"accounts: {', '.join(status['accounts']) if status['accounts'] else '(none)'}")
    _emit(f"active: {status['active'] or '(none)'}")
    _emit(f"bookmarks: {bookmarks_path() if bookmarks_available() else 'missing'} (SOPHON_BOOKMARKS_PATH)")
    searx_url = searxng_base_url() or "-"
    _emit(
        f"web search: {'on' if web_search_tools_enabled() else 'off'} "
        f"(searxng={'yes' if searxng_configured() else 'no'} url={searx_url} "
        f"cse={'yes' if cse_configured() else 'no'} "
        f"SOPHON_SEARXNG_URL / SOPHON_GOOGLE_CSE_KEY / SOPHON_GOOGLE_CSE_CX)"
    )
    return False


@_register("overleaf-status", "Show Overleaf Git token / project tool status.")
def _cmd_overleaf_status(state: _SessionState, _arg: str) -> bool:
    _ = state
    from integrations.overleaf.client import (
        configured_project_ids,
        overleaf_available,
        overleaf_git_token,
        overleaf_tools_enabled,
        ping,
    )

    enabled = overleaf_tools_enabled()
    status = ping()
    _emit(f"overleaf tools: {'on' if enabled else 'off'} (SOPHON_OVERLEAF_TOOLS)")
    _emit(f"token: {'set' if overleaf_git_token() else 'missing'} (SOPHON_OVERLEAF_GIT_TOKEN)")
    ids = configured_project_ids()
    _emit(f"projects: {', '.join(ids) if ids else '(none)'}")
    aliases = status.get("aliases") or {}
    if aliases:
        bits = [f"{alias}={pid}" for alias, pid in aliases.items()]
        _emit(f"aliases: {', '.join(bits)}")
    _emit(f"available: {'yes' if overleaf_available() else 'no'}")
    _emit(f"cache: {status.get('cache')}")
    if not enabled:
        _emit(
            "enable with SOPHON_OVERLEAF_TOOLS=1 plus SOPHON_OVERLEAF_GIT_TOKEN "
            "and SOPHON_OVERLEAF_PROJECT_ID (or PROJECT_IDS / data/overleaf/projects.json)."
        )
    _emit("slash: /overleaf-list [project_id] [path] /overleaf-read project_id path")
    return False


@_register("overleaf-list", "List Overleaf projects or files. /overleaf-list [project_id] [path].")
def _cmd_overleaf_list(state: _SessionState, arg: str) -> bool:
    _ = state
    from integrations.overleaf.client import format_tool_payload, list_files, list_projects

    parts = [bit for bit in arg.split() if bit]
    if not parts:
        rows = [
            {
                "project_id": project.project_id,
                "alias": project.alias,
                "path": str(project.path),
                "sync_error": project.sync_error,
            }
            for project in list_projects()
        ]
        _emit(format_tool_payload({"projects": rows}))
        return False
    project_id = parts[0]
    path = " ".join(parts[1:]) if len(parts) > 1 else ""
    try:
        entries = list_files(project_id, path)
    except Exception as exc:
        _emit(f"error: {exc}")
        return False
    _emit(
        format_tool_payload(
            {
                "project_id": project_id,
                "path": path or "/",
                "entries": [
                    {
                        "name": entry.name,
                        "path": entry.relpath,
                        "kind": "dir" if entry.is_dir else "file",
                    }
                    for entry in entries
                ],
            }
        )
    )
    return False


@_register("overleaf-read", "Read an Overleaf file. /overleaf-read PROJECT_ID PATH")
def _cmd_overleaf_read(state: _SessionState, arg: str) -> bool:
    _ = state
    from integrations.overleaf.client import read_file

    parts = [bit for bit in arg.split() if bit]
    if len(parts) < 2:
        _emit("usage: /overleaf-read PROJECT_ID PATH")
        return False
    project_id = parts[0]
    path = " ".join(parts[1:])
    try:
        _emit(read_file(project_id, path, truncate=True))
    except Exception as exc:
        _emit(f"error: {exc}")
    return False


@_register("tools", "Show enabled chat tools (speak / vault / zotero / google / overleaf / shell / editor / subagent).")
def _cmd_tools(state: _SessionState, _arg: str) -> bool:
    from cli.code_assist import editor_tools_enabled
    from integrations.google.bookmarks import bookmarks_available
    from integrations.google.oauth import active_email, google_tools_enabled, list_accounts
    from integrations.google.search import web_search_tools_enabled
    from integrations.obsidian.client import obsidian_tools_enabled
    from integrations.overleaf.client import overleaf_tools_enabled
    from integrations.shell.runner import ShellSession, shell_tools_enabled
    from integrations.zotero.client import zotero_tools_enabled
    from utils.device.env_bootstrap import DOTENV_LOAD_PATH

    if state.shell_session is None:
        state.shell_session = ShellSession(cwd=Path(state.project_root).resolve())
    session = state.shell_session
    assert isinstance(session, ShellSession)

    tts_on = _tts_tools_enabled()
    sst_on = _sst_tools_enabled()
    obs_on = obsidian_tools_enabled()
    zot_on = zotero_tools_enabled()
    google_on = _google_tools_wanted()
    overleaf_on = overleaf_tools_enabled()
    shell_on = shell_tools_enabled()
    editor_on = editor_tools_enabled()
    memory_on = state.memory_layer is not None and _memory_tools_wanted()
    skill_on = state.skill_catalog is not None and _skill_tools_wanted()
    spawn_on = _spawn_tools_in_schema(state)
    _emit(f"dotenv: {DOTENV_LOAD_PATH if DOTENV_LOAD_PATH else '(not loaded)'}")
    _emit(f"speak tool: {'on' if tts_on else 'off'} (SOPHON_TTS_TOOL)")
    _emit(f"transcribe tool: {'on' if sst_on else 'off'} (SOPHON_SST_TOOL)")
    _emit(f"vault tools: {'on' if obs_on else 'off'} (SOPHON_OBSIDIAN_TOOLS)")
    _emit(f"zotero tools: {'on' if zot_on else 'off'} (SOPHON_ZOTERO_TOOLS, default on if library reachable)")
    _emit(
        f"google tools: {'on' if google_on else 'off'} "
        f"(accounts={len(list_accounts())} active={active_email() or '-'} "
        f"bookmarks={'yes' if bookmarks_available() else 'no'} "
        f"search={'yes' if web_search_tools_enabled() else 'no'} SOPHON_GOOGLE_TOOLS / SOPHON_WEB_SEARCH_TOOLS / SOPHON_SEARXNG_URL)"
    )
    _emit(f"overleaf tools: {'on' if overleaf_on else 'off'} (SOPHON_OVERLEAF_TOOLS)")
    _emit(f"shell tools: {'on' if shell_on else 'off'} (SOPHON_SHELL_TOOLS)")
    _emit(f"editor tools: {'on' if editor_on else 'off'} (SOPHON_EDITOR_TOOLS)")
    from harness import ensure_harness

    harness = ensure_harness(state)
    _emit(f"harness mode: {harness.mode} (/mode plan|chat|agent)")
    _emit(f"harness workspace: {harness.workspace_path()}")
    _emit(f"memory tools: {'on' if memory_on else 'off'} (SOPHON_MEMORY_TOOLS)")
    _emit(f"skill tools: {'on' if skill_on else 'off'} (SOPHON_SKILL_TOOLS)")
    _emit(f"subagent tools: {'on' if spawn_on else 'off'} (SOPHON_SUBAGENT_TOOLS, omitted in plan/chat)")
    _emit(f"shell cwd: {session.cwd}")
    _emit(f"tool rounds: {_format_tool_max_rounds(state.tool_max_rounds)} (/tool-rounds, /unlimited)")
    if tts_on:
        _emit("speak: speak")
    if sst_on:
        _emit("sst: transcribe")
    if obs_on:
        _emit("vault: vault_search, vault_list, vault_read, vault_recent")
    if zot_on:
        _emit("zotero: zotero_tree, zotero_search, zotero_list, zotero_read, zotero_metrics")
    if google_on:
        bits: list[str] = []
        if google_tools_enabled() and list_accounts():
            bits.extend(["gmail_search", "gmail_read", "drive_tree", "drive_list"])
        if bookmarks_available():
            bits.append("bookmarks_tree")
        if web_search_tools_enabled():
            bits.append("web_search")
        _emit(f"google: {', '.join(bits) if bits else '(none)'}")
    if overleaf_on:
        _emit(
            "overleaf: overleaf_list_projects, overleaf_list, overleaf_read, overleaf_sections"
        )
    if shell_on:
        _emit("shell: shell_pwd, shell_cd, shell_ls, shell_read, shell_exec")
    if editor_on:
        _emit("editor: editor_read, editor_propose_edit, editor_status")
        _emit("review: /assist status | accept | decline | undo | redo")
    if memory_on:
        _emit("memory: memory_view, memory_search, memory_read, memory_write, memory_propose")
    if skill_on:
        _emit("skills: skill_list, skill_read, skill_read_file")
    if spawn_on:
        _emit("subagent: subagent, subagent_fork")
    if (
        not obs_on
        and not zot_on
        and not google_on
        and not overleaf_on
        and not shell_on
        and not tts_on
        and not sst_on
        and not editor_on
        and not memory_on
        and not skill_on
        and not spawn_on
    ):
        _emit("no chat tools enabled; set env flags in .env and restart")
    else:
        _emit("all enabled tool packs are available together (no tool-pack mode switch)")
        _emit("harness /mode plan denies shell_exec and proposes; /permissions lists rules")
    return False


@_register("subagents", "Last in-process subagent run (id, provider, depth, stop, output).")
def _cmd_subagents(state: _SessionState, _arg: str) -> bool:
    run = getattr(state, "last_subagent_run", None)
    if run is None:
        _emit("no subagent runs in this session")
        return False
    result = getattr(run, "result", None)
    output = str(getattr(result, "output", "") or "")
    if len(output) > 400:
        output = output[:399] + "…"
    stop = str(getattr(result, "stop_reason", "") or "")
    diag = str(getattr(result, "diagnostic", "") or "")
    _emit(f"id: {getattr(run, 'id', '')}")
    _emit(f"provider: {getattr(run, 'provider', '')}")
    _emit(f"depth: {getattr(run, 'depth', '')}")
    _emit(f"agent_type: {getattr(run, 'agent_type', '')}")
    _emit(f"label: {getattr(run, 'label', '')}")
    _emit(f"stop: {stop}")
    if diag:
        _emit(f"diagnostic: {diag}")
    _emit(output if output else "(empty)")
    return False


@_register("mode", "Harness mode: /mode [plan|chat|agent].")
def _cmd_mode(state: _SessionState, arg: str) -> bool:
    from harness import ensure_harness

    harness = ensure_harness(state)
    token = arg.strip().lower()
    if not token:
        _emit(f"mode: {harness.mode}")
        _emit("usage: /mode [plan|chat|agent]")
        return False
    try:
        mode = harness.set_mode(token)
    except ValueError as exc:
        _emit(f"error: {exc}")
        return False
    _emit(f"(harness mode = {mode})")
    return False


@_register("permissions", "Show harness rules. /permissions [reload].")
def _cmd_permissions(state: _SessionState, arg: str) -> bool:
    from harness import ensure_harness

    harness = ensure_harness(state)
    cmd = arg.strip().lower()
    if cmd in ("reload", "refresh"):
        harness.reload()
        _emit("(harness policy reloaded)")
    _emit(harness.describe())
    return False


@_register("assist", "Code review queue: /assist [status|accept|decline|undo|redo].")
def _cmd_assist(state: _SessionState, arg: str) -> bool:
    from cli.assist_tools import (
        accept_pending_edits,
        decline_pending_edits,
        redo_accepted_edits,
        undo_accepted_edits,
    )
    from cli.code_assist import ensure_assist

    controller = ensure_assist(state)
    cmd = arg.strip().lower() or "status"
    if cmd in ("status", "list", "show"):
        _emit(controller.status_text(state.editor_workspace or state.project_root))
        return False
    if cmd in ("accept", "apply"):
        _emit(accept_pending_edits(state))
        return False
    if cmd in ("decline", "reject", "dismiss"):
        _emit(decline_pending_edits(state))
        return False
    if cmd in ("undo",):
        _emit(undo_accepted_edits(state))
        return False
    if cmd in ("redo",):
        _emit(redo_accepted_edits(state))
        return False
    _emit("usage: /assist [status|accept|decline|undo|redo]")
    return False


@_register(
    "memory-status",
    "Show memory store and session info.",
    aliases=("memorystatus",),
)
def _cmd_memory_status(state: _SessionState, _arg: str) -> bool:
    if state.memory is None or state.memory_scope is None:
        _emit("memory: disabled (SOPHON_MEMORY_DB=0)")
        return False
    try:
        recent = state.memory.load_recent_turns(state.memory_scope, max(state.memory_recall_turns, 1))
    except Exception as exc:
        _emit(f"memory: error loading turns: {exc}")
        return False
    scope = state.memory_scope
    _emit(
        f"memory: session={scope.session_id} user={scope.user_id} "
        f"recall_turns={state.memory_recall_turns} loaded={len(recent)}"
    )
    if state.memory_layer is not None:
        status = state.memory_layer.status()
        _emit(
            "  tiers: "
            f"semantic={status['semantic']} procedural={status['procedural']} "
            f"working={status['working']} episodic={status['episodic']} "
            f"(last_dropped={state.last_memory_dropped})"
        )
    return False


@_register("tts", "TTS playback after replies: /tts on | off | show.")
def _cmd_tts(state: _SessionState, arg: str) -> bool:
    a = arg.strip().lower()
    if a in ("on", "1", "true", "yes"):
        state.tts_enabled = True
        _emit("(tts enabled)")
    elif a in ("off", "0", "false", "no"):
        state.tts_enabled = False
        _emit("(tts disabled)")
    elif a in ("show", "") or not arg.strip():
        ins = repr(state.tts_instruct) if state.tts_instruct else "(none)"
        om = repr(state.tts_ollama_model) if state.tts_ollama_model else "(none)"
        _emit(
            f"tts: {'on' if state.tts_enabled else 'off'} backend={state.tts_backend_id!r} "
            f"speaker={state.tts_speaker!r} lang={state.tts_language!r} "
            f"speed={state.tts_speed:g} plain={state.tts_plain_text} "
            f"max_chars={state.tts_max_chars} hf_model={state.tts_model_id!r} "
            f"ollama_model={om} instruct={ins} tool={'on' if _tts_tools_enabled() else 'off'}"
        )
    else:
        _emit("usage: /tts on | off | show")
    return False


@_register("speak", "Speak text now: /speak TEXT  (or /speak-file / /speak-last / /tts-test).")
def _cmd_speak(state: _SessionState, arg: str) -> bool:
    text = arg.strip()
    if not text:
        _emit("usage: /speak TEXT | /speak-file PATH | /speak-last [user|assistant] | /tts-test ...")
        return False
    _run_speak_request(state, text=text)
    return False


@_register("speak-file", "Speak a workspace document: /speak-file PATH.")
def _cmd_speak_file(state: _SessionState, arg: str) -> bool:
    path = arg.strip().strip("\"'")
    if not path:
        _emit("usage: /speak-file PATH")
        return False
    _run_speak_request(state, path=path)
    return False


@_register("speak-last", "Speak last chat turn: /speak-last [assistant|user].")
def _cmd_speak_last(state: _SessionState, arg: str) -> bool:
    role = arg.strip().lower() or "assistant"
    if role in ("a", "assistant", "last"):
        turn = "assistant"
    elif role in ("u", "user"):
        turn = "user"
    else:
        _emit("usage: /speak-last [assistant|user]")
        return False
    _run_speak_request(state, turn=turn)
    return False


@_register(
    "tts-test",
    "Test TTS playback: /tts-test last | user | turn N | file PATH | TEXT.",
)
def _cmd_tts_test(state: _SessionState, arg: str) -> bool:
    raw = arg.strip()
    if not raw:
        _emit(
            "usage: /tts-test last | user | turn <N> | file <PATH> | <literal text>\n"
            "plays via terminal audio (winsound/afplay/paplay)."
        )
        return False
    low = raw.lower()
    if low in ("last", "assistant"):
        _run_speak_request(state, turn="assistant")
        return False
    if low == "user":
        _run_speak_request(state, turn="user")
        return False
    if low.startswith("turn "):
        idx = raw.split(maxsplit=1)[1].strip()
        _run_speak_request(state, turn=idx)
        return False
    if low.startswith("file "):
        path = raw.split(maxsplit=1)[1].strip().strip("\"'")
        _run_speak_request(state, path=path)
        return False
    maybe_path = Path(raw.strip("\"'"))
    roots = _speak_roots(state)
    for root in roots:
        cand = maybe_path if maybe_path.is_absolute() else (root / maybe_path)
        if cand.is_file():
            _run_speak_request(state, path=str(maybe_path))
            return False
    _run_speak_request(state, text=raw)
    return False


@_register("tts-speaker", "Set TTS voice/speaker (Kokoro e.g. am_adam, am_michael, af_heart; Qwen e.g. Ryan).")
def _cmd_tts_speaker(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"tts speaker = {state.tts_speaker!r}")
        return False
    state.tts_speaker = arg.strip()
    _emit(f"(tts speaker = {state.tts_speaker!r})")
    return False


@_register("tts-speed", "Set TTS playback rate (0.5-2.0). Empty shows current.")
def _cmd_tts_speed(state: _SessionState, arg: str) -> bool:
    raw = arg.strip()
    if not raw:
        _emit(f"tts speed = {state.tts_speed:g}")
        return False
    try:
        value = float(raw)
    except ValueError:
        _emit("usage: /tts-speed <float 0.5-2.0>")
        return False
    if value < 0.5 or value > 2.0:
        _emit("tts speed must be between 0.5 and 2.0")
        return False
    state.tts_speed = value
    _emit(f"(tts speed = {state.tts_speed:g})")
    return False


@_register("tts-max-chars", "Set max characters spoken per TTS call. Empty shows current.")
def _cmd_tts_max_chars(state: _SessionState, arg: str) -> bool:
    raw = arg.strip()
    if not raw:
        _emit(f"tts max_chars = {state.tts_max_chars}")
        return False
    try:
        value = int(raw)
    except ValueError:
        _emit("usage: /tts-max-chars <positive int>")
        return False
    if value < 1:
        _emit("tts max_chars must be >= 1")
        return False
    state.tts_max_chars = value
    _emit(f"(tts max_chars = {state.tts_max_chars})")
    return False


@_register("tts-lang", "Set TTS language label (e.g. English, Chinese).")
def _cmd_tts_lang(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"tts language = {state.tts_language!r}")
        return False
    state.tts_language = arg.strip()
    _emit(f"(tts language = {state.tts_language!r})")
    return False


@_register("tts-instruct", "Set style instruct for TTS, or: /tts-instruct clear.")
def _cmd_tts_instruct(state: _SessionState, arg: str) -> bool:
    a = arg.strip()
    if a.lower() in ("", "clear", "none"):
        state.tts_instruct = None
        _emit("(tts instruct cleared)")
        return False
    state.tts_instruct = a
    _emit("(tts instruct updated)")
    return False


@_register("tts-plain", "Strip markdown fences before TTS: /tts-plain on | off | show.")
def _cmd_tts_plain(state: _SessionState, arg: str) -> bool:
    a = arg.strip().lower()
    if a in ("on", "1", "true", "yes"):
        state.tts_plain_text = True
        _emit("(tts plain = on)")
    elif a in ("off", "0", "false", "no"):
        state.tts_plain_text = False
        _emit("(tts plain = off)")
    elif a in ("show", "") or not arg.strip():
        _emit(f"tts plain = {'on' if state.tts_plain_text else 'off'}")
    else:
        _emit("usage: /tts-plain on | off | show")
    return False


@_register("tts-model", "Set CustomVoice HF repo id (reloads engine on next synthesis).")
def _cmd_tts_model(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"tts model = {state.tts_model_id!r}")
        return False
    state.tts_model_id = arg.strip()
    state.tts_engine = None
    _emit("(tts model updated)")
    return False


@_register(
    "tts-backend",
    f"TTS backend: /tts-backend {tts_backend_help_tokens()} | show.",
)
def _cmd_tts_backend(state: _SessionState, arg: str) -> bool:
    a = arg.strip()
    if a.lower() in ("show", "") or not arg.strip():
        _emit(f"tts backend = {state.tts_backend_id!r} ({tts_backend_help_tokens()})")
        return False
    nb = normalize_tts_backend_token(a)
    if nb is None:
        _emit(f"usage: /tts-backend {tts_backend_help_tokens()} | show")
        return False
    state.tts_backend_id = nb
    state.tts_engine = None
    _emit(f"(tts backend = {state.tts_backend_id!r})")
    return False


@_register("tts-ollama-model", "Set Ollama model tag for TTS when backend is ollama (clear | none to unset).")
def _cmd_tts_ollama_model(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"tts ollama model = {state.tts_ollama_model!r}")
        return False
    raw = arg.strip()
    if raw.lower() in ("clear", "none"):
        state.tts_ollama_model = None
    else:
        state.tts_ollama_model = raw
    state.tts_engine = None
    _emit("(tts ollama model updated)")
    return False


@_register("sst", "Qwen SST: /sst on | off | show. Dropped audio transcribes when on.")
def _cmd_sst(state: _SessionState, arg: str) -> bool:
    a = arg.strip().lower()
    if a in ("on", "1", "true", "yes"):
        state.sst_enabled = True
        _emit("(sst enabled)")
    elif a in ("off", "0", "false", "no"):
        state.sst_enabled = False
        _emit("(sst disabled)")
    elif a in ("show", "") or not arg.strip():
        lang = repr(state.sst_language) if state.sst_language else "auto"
        _emit(
            f"sst: {'on' if state.sst_enabled else 'off'} backend={state.sst_backend_id!r} "
            f"hf_model={state.sst_model_id!r} lang={lang} device={state.sst_device!r} "
            f"max_new_tokens={state.sst_max_new_tokens} tool={'on' if _sst_tools_enabled() else 'off'}"
        )
    else:
        _emit("usage: /sst on | off | show")
    return False


@_register("transcribe", "Transcribe an audio file with Qwen ASR: /transcribe PATH.")
def _cmd_transcribe(state: _SessionState, arg: str) -> bool:
    path = arg.strip().strip("\"'")
    if not path:
        _emit("usage: /transcribe PATH")
        return False
    _run_transcribe_request(state, path)
    return False


@_register("listen", "Toggle mic recording into the prompt: /listen [stop|cancel]. TUI: Speak button or Ctrl+L.")
def _cmd_listen(state: _SessionState, arg: str) -> bool:
    token = arg.strip().lower()
    if token in ("cancel", "discard"):
        cancel_sst_recording(state)
        return False
    if token in ("stop", "end"):
        finish_sst_recording(state)
        return False
    if sst_recording(state):
        finish_sst_recording(state)
        return False
    max_seconds: float | None = None
    if token:
        try:
            max_seconds = float(token)
        except ValueError:
            _emit("usage: /listen [stop|cancel]")
            return False
        max_seconds = max(2.0, min(max_seconds, 300.0))
    start_sst_recording(state, max_seconds=max_seconds)
    return False


@_register("listen-cancel", "Discard the current microphone recording.")
def _cmd_listen_cancel(state: _SessionState, _arg: str) -> bool:
    cancel_sst_recording(state)
    return False


@_register("play", "Replay a stored speech clip: /play [last|user|assistant|N] [1x|1.5x|2x].")
def _cmd_play(state: _SessionState, arg: str) -> bool:
    play_speech_clip(state, arg)
    return False


@_register("clips", "List stored speech clips for this chat session.")
def _cmd_clips(state: _SessionState, _arg: str) -> bool:
    store = _ensure_speech_store(state)
    if not store.clips:
        _emit(f"(no speech clips in {store.root})")
        return False
    _emit(f"speech clips ({len(store.clips)}) · {store.root}")
    for clip in store.clips:
        who = "you" if clip.role == "user" else "assistant"
        _emit(
            f"  {clip.clip_id:>3}  {who:<9} {clip.source:<4} {clip.duration_s:.1f}s  "
            f"{clip.path.name}"
        )
    speeds = " ".join(format_replay_speed(s) for s in default_replay_speeds())
    _emit(f"replay: /play N [{speeds}]")
    return False


@_register("sst-backend", f"SST backend: /sst-backend {stt_backend_help_tokens()} | show.")
def _cmd_sst_backend(state: _SessionState, arg: str) -> bool:
    a = arg.strip()
    if a.lower() in ("show", "") or not arg.strip():
        _emit(f"sst backend = {state.sst_backend_id!r} ({stt_backend_help_tokens()})")
        return False
    nb = normalize_stt_backend_token(a)
    if nb is None:
        _emit(f"usage: /sst-backend {stt_backend_help_tokens()} | show")
        return False
    state.sst_backend_id = nb
    state.sst_engine = None
    _emit(f"(sst backend = {state.sst_backend_id!r})")
    return False


@_register("sst-model", "Set Qwen ASR Hub repo id (reloads engine on next transcribe).")
def _cmd_sst_model(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"sst model = {state.sst_model_id!r}")
        return False
    state.sst_model_id = arg.strip()
    state.sst_engine = None
    _emit("(sst model updated)")
    return False


@_register("sst-lang", "Set SST language (English, German, auto). Empty shows current.")
def _cmd_sst_lang(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        lang = repr(state.sst_language) if state.sst_language else "auto"
        _emit(f"sst language = {lang}")
        return False
    state.sst_language = normalize_stt_language(arg)
    _emit(f"(sst language = {state.sst_language!r})" if state.sst_language else "(sst language = auto)")
    return False


def _queue_memory_promotion(state: _SessionState, ids: list[str]) -> str:
    layer = state.memory_layer
    if layer is None:
        return "memory: layer disabled (SOPHON_MEMORY_DB=0)"
    proposal = layer.propose_promotion(ids)
    if not proposal.promoted_ids:
        skipped = f" skipped: {', '.join(proposal.skipped_ids)}" if proposal.skipped_ids else ""
        return f"memory propose: nothing to promote.{skipped}"
    from cli.code_assist import FileEdit, ensure_assist, notify_assist_ui

    controller = ensure_assist(state)
    edit = FileEdit(
        path=proposal.path,
        before=proposal.before,
        after=proposal.after,
        description=f"memory: promote {', '.join(proposal.promoted_ids)}",
    )
    controller.add_edit(edit)
    notify_assist_ui(state)
    msg = (
        f"queued promotion of {', '.join(proposal.promoted_ids)} to facts.md. "
        "Accept or Decline in Review."
    )
    if proposal.skipped_ids:
        msg += f" skipped: {', '.join(proposal.skipped_ids)}"
    return msg


def _memory_status_lines(state: _SessionState) -> list[str]:
    lines: list[str] = []
    scope = state.memory_scope
    if scope is not None:
        lines.append(f"session={scope.session_id} user={scope.user_id}")
    if state.memory_layer is not None:
        status = state.memory_layer.status()
        lines.append(
            "tiers: "
            f"semantic={status['semantic']} procedural={status['procedural']} "
            f"working={status['working']} episodic={status['episodic']}"
        )
    if state.memory_budget is not None:
        lines.append(
            f"budget: {state.memory_budget.total_chars} chars, "
            f"recall_turns={state.memory_budget.recall_turns}, "
            f"last_dropped={state.last_memory_dropped}"
        )
    return lines


@_register("memory", "Memory layer: status|list|search|note|promote|revoke|compact.")
def _cmd_memory(state: _SessionState, arg: str) -> bool:
    layer = state.memory_layer
    parts = arg.strip().split(maxsplit=1)
    sub = parts[0].lower() if parts else "status"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if layer is None:
        _emit("memory: layer disabled (SOPHON_MEMORY_DB=0)")
        return False

    if sub in ("status", ""):
        for line in _memory_status_lines(state):
            _emit(line)
        return False

    if sub == "list":
        from processing.text.memory import normalize_tier

        tier = normalize_tier(rest or "semantic")
        entries = layer.list(tier)
        if not entries:
            _emit(f"memory {tier}: (none)")
            return False
        _emit(f"memory {tier}:")
        for entry in entries:
            _emit(f"  {entry.id} [{entry.confidence}] {entry.text}")
        return False

    if sub == "search":
        if not rest:
            _emit("usage: /memory search <query>")
            return False
        hits = layer.search(rest, limit=15)
        if not hits:
            _emit("memory search: (no hits)")
            return False
        for entry in hits:
            _emit(f"  [{entry.tier}] {entry.id} [{entry.confidence}] {entry.text}")
        return False

    if sub == "note":
        if not rest:
            _emit("usage: /memory note <text>")
            return False
        _, message = layer.note(rest, confidence="decided")
        _emit(f"(memory {message})")
        return False

    if sub == "promote":
        ids = rest.split()
        if not ids:
            _emit("usage: /memory promote <id> [id...]")
            return False
        _emit(f"(memory {_queue_memory_promotion(state, ids)})")
        return False

    if sub == "revoke":
        if not rest:
            _emit("usage: /memory revoke <fact-id>")
            return False
        ok, message = layer.revoke(rest.split()[0])
        _emit(f"(memory {message})" if ok else f"memory: {message}")
        return False

    if sub == "compact":
        _emit(f"(memory {layer.compact()})")
        return False

    _emit("usage: /memory [status|list <tier>|search <q>|note <text>|promote <id>|revoke <id>|compact]")
    return False


@_register("memory-clear", "Wipe persistent memory for the current session.")
def _cmd_memory_clear(state: _SessionState, _arg: str) -> bool:
    if state.memory is None or state.memory_scope is None:
        _emit("memory: disabled")
        return False
    try:
        removed = state.memory.clear_scope(state.memory_scope)
    except Exception as exc:
        _emit(f"memory: clear failed: {exc}")
        return False
    state._last_persisted_user_obj_id = 0
    _emit(f"(memory cleared, {removed} rows removed)")
    return False


def _queue_skill_proposal(state: _SessionState, proposal) -> str:
    from cli.code_assist import FileEdit, ensure_assist, notify_assist_ui

    controller = ensure_assist(state)
    verb = "create" if not proposal.before else "update"
    edit = FileEdit(
        path=proposal.path,
        before=proposal.before,
        after=proposal.after,
        description=f"skill: {verb} {proposal.name}",
    )
    controller.add_edit(edit)
    notify_assist_ui(state)
    return f"queued {verb} of skill {proposal.name}. Accept or Decline in Review."


@_register(
    "skill",
    "Skills: list|show <name>|attach <name>|detach [name]|create <name>|import <path>.",
    aliases=("skills",),
)
def _cmd_skill(state: _SessionState, arg: str) -> bool:
    catalog = state.skill_catalog
    if catalog is None:
        _emit("skills: disabled (SOPHON_SKILLS=0)")
        return False
    parts = arg.strip().split(maxsplit=1)
    sub = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("list", ""):
        catalog.refresh()
        if not catalog.entries:
            _emit("skills: (none found)")
            return False
        _emit("skills:")
        for entry in catalog.entries:
            mark = " *attached" if entry.name in state.attached_skills else ""
            desc = " ".join(entry.description.split())
            _emit(f"  {entry.name} root={entry.root_label}{mark}: {desc}")
        if catalog.shadowed:
            _emit("  shadowed (not used):")
            for name, root, winner in catalog.shadowed:
                _emit(f"    {name} root={root} hidden by root={winner}")
        return False

    if sub == "show":
        if not rest:
            _emit("usage: /skill show <name>")
            return False
        entry = catalog.get(rest)
        if entry is None:
            _emit(f"skills: no skill named {rest!r}")
            return False
        _emit(f"# {entry.name} ({entry.root_label})")
        _emit(entry.description)
        body = entry.spec.body.strip()
        if body:
            _emit("")
            _emit(body)
        return False

    if sub == "attach":
        if not rest:
            _emit("usage: /skill attach <name>")
            return False
        entry = catalog.get(rest)
        if entry is None:
            _emit(f"skills: no skill named {rest!r}")
            return False
        if entry.name not in state.attached_skills:
            state.attached_skills.append(entry.name)
        _emit(f"(skill attached: {entry.name})")
        return False

    if sub == "detach":
        if not rest:
            count = len(state.attached_skills)
            state.attached_skills.clear()
            _emit(f"(skills detached: {count})")
            return False
        target = rest.strip().lower()
        if target in state.attached_skills:
            state.attached_skills.remove(target)
            _emit(f"(skill detached: {target})")
        else:
            _emit(f"skills: {target!r} was not attached")
        return False

    if sub == "create":
        tokens = rest.split()
        user = "--user" in tokens
        tokens = [t for t in tokens if t != "--user"]
        if not tokens:
            _emit("usage: /skill create <name> [--user] [description]")
            return False
        name = tokens[0]
        description = " ".join(tokens[1:]).strip() or None
        from processing.text.skills import scaffold_skill

        try:
            proposal = scaffold_skill(state.project_root, name, description, user=user)
        except ValueError as exc:
            _emit(f"skills: {exc}")
            return False
        _emit(f"(skill {_queue_skill_proposal(state, proposal)})")
        return False

    if sub == "import":
        tokens = rest.split()
        user = "--user" in tokens
        tokens = [t for t in tokens if t != "--user"]
        if not tokens:
            _emit("usage: /skill import <path> [--user]")
            return False
        source = " ".join(tokens).strip()
        from processing.text.skills import import_skill

        try:
            proposal = import_skill(state.project_root, source, user=user)
        except ValueError as exc:
            _emit(f"skills: {exc}")
            return False
        _emit(f"(skill {_queue_skill_proposal(state, proposal)})")
        return False

    _emit("usage: /skill [list|show <name>|attach <name>|detach [name]|create <name>|import <path>]")
    return False


def _parse_float(arg: str) -> float | None:
    try:
        return float(arg.strip())
    except (TypeError, ValueError):
        return None


def _parse_int(arg: str) -> int | None:
    try:
        return int(arg.strip())
    except (TypeError, ValueError):
        return None


@_register("params", "Show generation params (temp/top-p/max). Change with /temp /top-p /max /seed /rep /tool-rounds.")
def _cmd_params(state: _SessionState, _arg: str) -> bool:
    p = state.params
    _emit(
        f"params: max_new_tokens={p.max_new_tokens} temperature={p.temperature} "
        f"top_p={p.top_p} top_k={p.top_k} repetition_penalty={p.repetition_penalty} seed={p.seed} "
        f"tool_max_rounds={_format_tool_max_rounds(state.tool_max_rounds)}"
    )
    if is_server_backend(state.backend_id):
        _emit(
            f"context: {state.backend_id} context window is set when the model is loaded "
            f"in the server UI (not via sophon per request). Reply budget is /max."
        )
    elif state.meta is not None and state.meta.max_position_embeddings:
        _emit(f"context window: {state.meta.max_position_embeddings:,} tokens (from local weights)")
    else:
        _emit("context window: unknown")
    return False


@_register("temp", "Set sampling temperature (float). Empty arg shows current value.")
def _cmd_temp(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"temperature = {state.params.temperature}")
        return False
    v = _parse_float(arg)
    if v is None:
        _emit("usage: /temp <float>")
        return False
    state.params.temperature = v
    _emit(f"(temperature = {v})")
    return False


@_register("top-p", "Set top_p (float in (0,1]).")
def _cmd_topp(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"top_p = {state.params.top_p}")
        return False
    v = _parse_float(arg)
    if v is None:
        _emit("usage: /top-p <float>")
        return False
    state.params.top_p = v
    _emit(f"(top_p = {v})")
    return False


@_register("top-k", "Set top_k (positive int).")
def _cmd_topk(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"top_k = {state.params.top_k}")
        return False
    v = _parse_int(arg)
    if v is None or v <= 0:
        _emit("usage: /top-k <positive int>")
        return False
    state.params.top_k = v
    _emit(f"(top_k = {v})")
    return False


@_register("max", "Set max_new_tokens per reply (positive int).")
def _cmd_max(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"max_new_tokens = {state.params.max_new_tokens}")
        return False
    v = _parse_int(arg)
    if v is None or v <= 0:
        _emit("usage: /max <positive int>")
        return False
    state.params.max_new_tokens = v
    _emit(f"(max_new_tokens = {v})")
    return False


@_register("seed", "Set generation seed (int) or empty for current.")
def _cmd_seed(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"seed = {state.params.seed}")
        return False
    v = _parse_int(arg)
    if v is None:
        _emit("usage: /seed <int>")
        return False
    state.params.seed = v
    _emit(f"(seed = {v})")
    return False


@_register("rep", "Set repetition_penalty (float, >=1 discourages repeat).")
def _cmd_rep(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"repetition_penalty = {state.params.repetition_penalty}")
        return False
    v = _parse_float(arg)
    if v is None:
        _emit("usage: /rep <float>")
        return False
    state.params.repetition_penalty = v
    _emit(f"(repetition_penalty = {v})")
    return False


@_register("tool-rounds", "Set LM Studio tool-loop rounds: /tool-rounds <N | unlimited>. Empty shows current.")
def _cmd_tool_rounds(state: _SessionState, arg: str) -> bool:
    token = arg.strip().lower()
    if not token:
        _emit(f"tool_max_rounds = {_format_tool_max_rounds(state.tool_max_rounds)}")
        return False
    if token in ("unlimited", "inf", "infinite", "none", "0"):
        state.tool_max_rounds = None
        _emit("(tool_max_rounds = unlimited)")
        return False
    v = _parse_int(token)
    if v is None or v < 1:
        _emit("usage: /tool-rounds <positive int | unlimited>")
        return False
    state.tool_max_rounds = v
    _emit(f"(tool_max_rounds = {v})")
    return False


@_register("unlimited", "Remove the tool-loop round cap (same as /tool-rounds unlimited).")
def _cmd_unlimited(state: _SessionState, _arg: str) -> bool:
    _ = _arg
    state.tool_max_rounds = None
    _emit("(tool_max_rounds = unlimited)")
    return False


@_register("chat", "Leave shell mode and return to chat.")
def _cmd_chat_mode(state: _SessionState, _arg: str) -> bool:
    _ = _arg
    state.shell_mode = False
    _emit("(shell mode off)")
    return False


@_register("auto-attach", "Attach @path file text on send: /auto-attach on | off | show.")
def _cmd_auto_attach(state: _SessionState, arg: str) -> bool:
    token = arg.strip().lower()
    if token in ("", "show"):
        _emit("auto-attach = " + ("on" if state.auto_attach else "off"))
        return False
    if token in ("on", "1", "true", "yes"):
        state.auto_attach = True
    elif token in ("off", "0", "false", "no"):
        state.auto_attach = False
    else:
        _emit("usage: /auto-attach on | off | show")
        return False
    _emit("(auto-attach = " + ("on" if state.auto_attach else "off") + ")")
    return False


def prepare_user_message_text(state: _SessionState, line: str) -> str:
    from cli.tui.paste_drop import expand_at_paths, mention_roots

    raw = line.rstrip()
    if "\n--- file:" in raw or "\n--- path:" in raw:
        return raw
    extra: list[Path] = []
    root = getattr(state, "project_root", None)
    if root is not None:
        extra.append(Path(root))
    return expand_at_paths(
        raw,
        mention_roots(extra),
        attach=bool(getattr(state, "auto_attach", True)),
    )


def trim_incomplete_sentence(text: str, stop_reason: str) -> str:
    reason = str(stop_reason or "").lower()
    if reason not in ("length", "max_new_tokens", "max_tokens"):
        return text
    stripped = (text or "").rstrip()
    if not stripped:
        return text
    if stripped[-1] in ".?!…":
        return text
    last = max(stripped.rfind("."), stripped.rfind("?"), stripped.rfind("!"))
    if last < 0:
        return text
    return stripped[: last + 1]


def _run_shell_mode_command(state: _SessionState, command: str) -> None:
    from types import SimpleNamespace

    from integrations.shell.tools import SHELL_EXEC

    call = SimpleNamespace(name=SHELL_EXEC, arguments={"command": command})
    result = _execute_one_tool(state, call)
    _emit(result)


def _dispatch_command(state: _SessionState, line: str) -> tuple[bool, bool]:
    """Returns (handled, should_generate)."""
    if not line:
        return True, False
    stripped = line.strip()
    if stripped.lower() in ("exit", "quit"):
        state.exit_requested = True
        return True, False

    if stripped == "!":
        state.shell_mode = not bool(getattr(state, "shell_mode", False))
        _emit("(shell mode on)" if state.shell_mode else "(shell mode off)")
        return True, False

    if len(stripped) == 1 and stripped in _SHORTCUTS:
        name = _SHORTCUTS[stripped]
        return True, _COMMANDS[name].handler(state, "")
    if len(stripped) == 1 and stripped.lower() in _SHORTCUTS:
        name = _SHORTCUTS[stripped.lower()]
        return True, _COMMANDS[name].handler(state, "")

    if stripped.startswith("/"):
        rest = stripped[1:]
        if not rest:
            _emit("(empty command)")
            return True, False
        parts = rest.split(maxsplit=1)
        name = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        cmd = _COMMANDS.get(name)
        if cmd is None:
            _emit(f"(unknown command: /{name})  type /help")
            return True, False
        from cli.slash_index import remember_slash_use

        state.slash_recents = remember_slash_use(list(state.slash_recents), cmd.name)
        return True, cmd.handler(state, arg)

    if bool(getattr(state, "shell_mode", False)):
        _run_shell_mode_command(state, line.rstrip())
        return True, False

    return False, False


def _latest_user_text(state: _SessionState) -> str | None:
    for msg in reversed(state.messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            return str(content) if content is not None else None
    return None


def _record_retrieval_metrics(state: _SessionState, result: RetrievalResult) -> None:
    raw = result.extras.get("metrics") if isinstance(result.extras, dict) else None
    if not isinstance(raw, dict):
        return
    state.last_retrieval_metrics = raw
    hist = state.retrieval_probe_history
    hist.append(raw)
    if len(hist) > 64:
        del hist[:-64]
    if state.debug:
        from processing.text.retrieval.metrics import RetrievalQueryMetrics, format_query_metrics

        try:
            _emit(
                "(retrieval "
                + format_query_metrics(
                    RetrievalQueryMetrics(
                        latency_s=float(raw.get("latency_s") or 0.0),
                        n_hits=int(raw.get("n_hits") or 0),
                        top_k=int(raw.get("top_k") or state.retrieval_top_k),
                        query_chars=int(raw.get("query_chars") or 0),
                        top_score=raw.get("top_score"),
                        mean_score=raw.get("mean_score"),
                        min_score=raw.get("min_score"),
                        score_margin=raw.get("score_margin"),
                        backend_id=str(raw.get("backend_id") or ""),
                    )
                )
                + ")"
            )
        except Exception:
            _emit(f"(retrieval metrics={raw})")


def _run_retrieval(state: _SessionState) -> RetrievalResult | None:
    if state.retriever is None and state.structure_retriever is None:
        return None
    query_text = _latest_user_text(state)
    if not query_text:
        return None
    structure_available = state.structure_retriever is not None
    if state.rag_adaptive:
        adaptive = decide_retrieval(query_text, structure_available=structure_available)
    else:
        adaptive = AdaptiveDecision(RetrievalDecision.SINGLE_HOP, "adaptive_disabled")
    state.last_retrieval_decision = {
        "decision": adaptive.decision.value,
        "reason": adaptive.reason,
    }
    if adaptive.decision == RetrievalDecision.SKIP:
        if state.debug:
            _emit(f"(retrieval skipped: {adaptive.reason})")
        result = empty_result()
        result.extras["adaptive"] = state.last_retrieval_decision
        return result
    active: RagRetriever | None
    if adaptive.decision == RetrievalDecision.MULTI_HOP and state.structure_retriever is not None:
        active = state.structure_retriever
    else:
        active = state.retriever or state.structure_retriever
    if active is None:
        return None
    try:
        import time

        t0 = time.perf_counter()
        result = active.retrieve(
            RetrievalQuery(text=query_text, params={"top_k": state.retrieval_top_k}),
        )
        if not isinstance(result.extras.get("metrics"), dict):
            from processing.text.retrieval.metrics import build_query_metrics

            metrics = build_query_metrics(
                latency_s=time.perf_counter() - t0,
                chunks=list(result.chunks),
                top_k=state.retrieval_top_k,
                query_chars=len(query_text),
                backend_id=active.backend_id(),
            )
            result.extras["metrics"] = metrics.as_dict()
        result.extras["adaptive"] = state.last_retrieval_decision
        _record_retrieval_metrics(state, result)
        return result
    except Exception as exc:
        _emit(f"(retrieval failed: {exc})")
        return None


def _retrieval_trace_from_run(
    state: _SessionState,
    retrieval: RetrievalResult | None,
) -> RetrievalTrace:
    enabled = state.retriever is not None or state.structure_retriever is not None
    if not enabled:
        return RetrievalTrace(enabled=False)
    if retrieval is None:
        return RetrievalTrace(enabled=True, failed=True, reason="error")
    extras = retrieval.extras if isinstance(retrieval.extras, dict) else {}
    adaptive = extras.get("adaptive") if isinstance(extras.get("adaptive"), dict) else None
    if not isinstance(adaptive, dict):
        adaptive = state.last_retrieval_decision if isinstance(state.last_retrieval_decision, dict) else {}
    decision = str(adaptive.get("decision") or "")
    reason = str(adaptive.get("reason") or "")
    metrics = extras.get("metrics") if isinstance(extras.get("metrics"), dict) else None
    backend_id = ""
    if isinstance(metrics, dict):
        backend_id = str(metrics.get("backend_id") or "")
    if not backend_id:
        active = state.retriever or state.structure_retriever
        if active is not None:
            try:
                backend_id = str(active.backend_id())
            except Exception:
                backend_id = ""
    skipped = decision == RetrievalDecision.SKIP.value
    return RetrievalTrace(
        enabled=True,
        skipped=skipped,
        failed=False,
        decision=decision,
        reason=reason,
        backend_id=backend_id,
        metrics=metrics,
    )


def _completion_reasoning(completion: object) -> str | None:
    from cli.agent_runtime import completion_reasoning

    return completion_reasoning(completion)


@dataclass
class _ServerLoopOutcome:
    text: str
    elapsed_s: float
    spoke: bool
    prompt_tokens: int
    completion_tokens: int
    stop_reason: str
    reasoning: str | None
    tools: list[ToolCallTrace]
    rounds: int


def _commit_assistant_turn(
    state: _SessionState,
    text: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    elapsed: float,
    stop_reason: str,
    retrieval: RetrievalResult | None,
    spoke: bool,
    reasoning: str | None = None,
    plan_text: str | None = None,
    tools: list[ToolCallTrace] | None = None,
    tool_rounds: int = 0,
) -> None:
    text = trim_incomplete_sentence(text, stop_reason)
    state.messages.append({"role": "assistant", "content": text})
    _persist_turn_after_success(state)
    state.stats.record_turn(
        input_tokens=prompt_tokens,
        new_tokens=completion_tokens,
        gen_time_s=elapsed,
        stop_reason=stop_reason,
    )
    trace = build_turn_trace(
        input_tokens=prompt_tokens,
        new_tokens=completion_tokens,
        gen_time_s=elapsed,
        stop_reason=stop_reason,
        backend_id=str(state.backend_id),
        retrieval=_retrieval_trace_from_run(state, retrieval),
        reasoning=reasoning,
        plan_text=plan_text,
        tools=tools,
        tool_rounds=tool_rounds,
    )
    _emit_assistant(str(text), trace)
    if not spoke:
        _maybe_play_assistant_tts(state, str(text))
    if state.debug:
        _emit(state.stats.format_footer())


def _load_memory_turns(state: _SessionState) -> list[object]:
    if state.memory is None or state.memory_scope is None or state.memory_recall_turns <= 0:
        return []
    try:
        return list(state.memory.load_recent_turns(state.memory_scope, state.memory_recall_turns))
    except Exception as exc:
        _emit(f"(memory load failed: {exc})")
        return []


def _last_user_text(state: _SessionState) -> str:
    for msg in reversed(state.messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _build_memory_pack(state: _SessionState):
    if state.memory_layer is None:
        return None
    from processing.text.memory import budget_from_env

    budget = state.memory_budget or budget_from_env(max(state.memory_recall_turns, 0))
    try:
        pack = state.memory_layer.pack(_last_user_text(state), budget)
    except Exception as exc:
        _emit(f"(memory pack failed: {exc})")
        return None
    state.last_memory_dropped = pack.total_dropped
    return pack


def _build_skills_block(state: _SessionState) -> str | None:
    catalog = state.skill_catalog
    if catalog is None:
        return None
    lines: list[str] = []
    if catalog.entries:
        cat_lines, dropped = catalog.catalog_lines()
        if cat_lines:
            lines.append("Skills (reusable procedures; read one before you follow it):")
            lines.extend(cat_lines)
            if dropped > 0:
                lines.append(f"(skills: {dropped} hidden for budget; use /skill list)")
    attached: list[str] = []
    for name in state.attached_skills:
        entry = catalog.get(name)
        if entry is None:
            continue
        body = entry.spec.body.strip()
        header = f"## Skill: {entry.name}"
        attached.append(f"{header}\n{body}" if body else header)
    if attached:
        lines.append("")
        lines.append("Attached skills (follow these now):")
        lines.append("\n\n".join(attached))
    if not lines:
        return None
    return "\n".join(lines).strip()


def _build_call_messages(
    state: _SessionState,
    retrieval: RetrievalResult | None,
) -> tuple[list[dict[str, object]], ContextBuildResult]:
    memory_pack = _build_memory_pack(state)
    memory_turns = [] if memory_pack is not None else _load_memory_turns(state)
    pack_has_content = memory_pack is not None and not memory_pack.is_empty()
    skills_block = _build_skills_block(state)
    if retrieval is None and not memory_turns and not pack_has_content and not skills_block:
        return list(state.messages), ContextBuildResult(messages=list(state.messages))
    base = [m for m in state.messages if m.get("role") != "system"]
    built = build_messages_for_model(
        base,
        retrieval=retrieval,
        memory_turns=memory_turns,
        memory_pack=memory_pack,
        skills_block=skills_block,
        system_text=state.system_text,
        max_retrieval_chunks=state.retrieval_top_k,
    )
    return built.messages, built


def _inject_token_budget(messages: list[dict[str, object]], max_new_tokens: int) -> list[dict[str, object]]:
    extra = (
        "Reply budget this turn is "
        + str(max(int(max_new_tokens), 1))
        + " tokens. End on a complete sentence. Do not start a sentence you cannot finish."
    )
    out = [dict(item) for item in messages]
    for item in out:
        if item.get("role") == "system":
            body = str(item.get("content") or "").rstrip()
            item["content"] = body + "\n" + extra if body else extra
            return out
    out.insert(0, {"role": "system", "content": extra})
    return out


def _persist_turn_after_success(state: _SessionState) -> None:
    if state.memory is None or state.memory_scope is None:
        return
    if not state.messages or state.messages[-1].get("role") != "assistant":
        return
    assistant_msg = state.messages[-1]
    user_msg: dict[str, object] | None = None
    if len(state.messages) >= 2 and state.messages[-2].get("role") == "user":
        user_msg = state.messages[-2]
    try:
        if user_msg is not None and id(user_msg) != state._last_persisted_user_obj_id:
            state.memory.append_turn(
                state.memory_scope,
                "user",
                str(user_msg.get("content", "")),
            )
            state._last_persisted_user_obj_id = id(user_msg)
        state.memory.append_turn(
            state.memory_scope,
            "assistant",
            str(assistant_msg.get("content", "")),
        )
    except Exception as exc:
        _emit(f"(memory persist failed: {exc})")


def _tts_engine_singleton(state: _SessionState) -> SpeechTtsEngine:
    if state.tts_engine is None:
        state.tts_engine = create_speech_tts_engine(
            state.tts_backend_id,
            hf_model_id=state.tts_model_id,
            hf_device_hint=state.tts_device,
            ollama_tts_model=state.tts_ollama_model,
        )
    return state.tts_engine


def _sst_engine_singleton(state: _SessionState) -> SpeechSttEngine:
    if state.sst_engine is None:
        state.sst_engine = create_speech_stt_engine(
            state.sst_backend_id,
            hf_model_id=state.sst_model_id,
            hf_device_hint=state.sst_device,
            max_new_tokens=max(state.sst_max_new_tokens, 1),
        )
    return state.sst_engine


def _speak_roots(state: _SessionState) -> list[Path]:
    roots = [Path(state.project_root).resolve(), Path.cwd().resolve()]
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _speech_session_id(state: _SessionState) -> str:
    if state.memory_scope is not None and str(state.memory_scope.session_id).strip():
        return str(state.memory_scope.session_id)
    return _dt.datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _ensure_speech_store(state: _SessionState) -> SessionSpeechStore:
    if state.speech_store is None:
        state.speech_store = SessionSpeechStore.open(_speech_session_id(state))
    return state.speech_store


def _make_speech_store(memory_scope: MemoryScope | None) -> SessionSpeechStore:
    if memory_scope is not None and str(memory_scope.session_id).strip():
        session_id = str(memory_scope.session_id)
    else:
        session_id = _dt.datetime.now().strftime("session_%Y%m%d_%H%M%S")
    return SessionSpeechStore.open(session_id)


def _speak_now(state: _SessionState, text: str) -> None:
    store = _ensure_speech_store(state)
    clip = speak_text_to_store(
        engine=_tts_engine_singleton(state),
        store=store,
        text=text,
        speaker=state.tts_speaker,
        language=state.tts_language,
        instruct=state.tts_instruct,
        max_chars=max(state.tts_max_chars, 1),
        emit=_emit,
        plain_text=state.tts_plain_text,
        speed=float(getattr(state, "tts_speed", 1.0) or 1.0),
        role="assistant",
    )
    if clip is not None:
        _announce_speech_clip(clip)
        return
    speak_text_blocking(
        engine=_tts_engine_singleton(state),
        text=text,
        speaker=state.tts_speaker,
        language=state.tts_language,
        instruct=state.tts_instruct,
        max_chars=max(state.tts_max_chars, 1),
        emit=_emit,
        plain_text=state.tts_plain_text,
        speed=float(getattr(state, "tts_speed", 1.0) or 1.0),
    )


def _resolve_play_clip(
    store: SessionSpeechStore,
    *,
    token: str | None,
) -> SpeechClip | None:
    if token is None or token == "":
        return store.last()
    low = token.lower()
    if low in ("last", "latest"):
        return store.last()
    if low in ("user", "you", "me", "mic"):
        return store.last(role="user")
    if low in ("assistant", "model", "tts"):
        return store.last(role="assistant")
    try:
        clip_id = int(token)
    except ValueError:
        return None
    return store.get(clip_id)


def play_speech_clip(state: _SessionState, arg: str = "", speed: float | None = None) -> None:
    store = _ensure_speech_store(state)
    raw = arg.strip()
    tokens = raw.split() if raw else []
    clip_token: str | None = None
    rate = speed
    if tokens:
        first = tokens[0]
        parsed_speed = parse_replay_speed(first)
        first_is_clip = first.isdigit() or first.lower() in (
            "last",
            "latest",
            "user",
            "you",
            "me",
            "mic",
            "assistant",
            "model",
            "tts",
        )
        if len(tokens) == 1:
            if first_is_clip:
                clip_token = first
            elif parsed_speed is not None:
                rate = parsed_speed
            else:
                _emit("usage: /play [last|user|assistant|N] [1x|1.5x|2x]")
                return
        else:
            clip_token = first
            parsed_speed = parse_replay_speed(tokens[1])
            if parsed_speed is None:
                _emit("usage: /play [last|user|assistant|N] [1x|1.5x|2x]")
                return
            rate = parsed_speed
    clip = _resolve_play_clip(store, token=clip_token)
    if clip is None:
        _emit("(no matching speech clip)")
        return
    if not clip.path.is_file():
        _emit(f"(clip {clip.clip_id} missing: {clip.path})")
        return
    if rate is None:
        rate = 1.0
    who = "you" if clip.role == "user" else "assistant"
    _emit(f"(playing clip {clip.clip_id} {who} {format_replay_speed(rate)})")
    try:
        play_wav_file(clip.path, speed=rate)
    except Exception as exc:
        _emit(f"(play failed: {exc})")


def _run_speak_request(
    state: _SessionState,
    *,
    text: str | None = None,
    path: str | None = None,
    turn: str | None = None,
) -> None:
    from processing.audio.speech.sources import resolve_speak_payload

    try:
        body, source = resolve_speak_payload(
            text=text,
            path=path,
            turn=turn,
            messages=state.messages,
            roots=_speak_roots(state),
        )
    except Exception as exc:
        _emit(f"(speak failed: {exc})")
        return
    _emit(f"(speak {source}, {len(body)} chars)")
    _speak_now(state, body)


def _maybe_play_assistant_tts(state: _SessionState, text: str) -> None:
    if not state.tts_enabled:
        return
    _speak_now(state, text)


def _tts_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_TTS_TOOL", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _sst_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_SST_TOOL", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _obsidian_tools_wanted() -> bool:
    from integrations.obsidian.client import obsidian_tools_enabled

    return obsidian_tools_enabled()


def _zotero_tools_wanted() -> bool:
    from integrations.zotero.client import zotero_tools_enabled

    return zotero_tools_enabled()


def _google_tools_wanted() -> bool:
    from integrations.google.bookmarks import bookmarks_available
    from integrations.google.oauth import google_tools_enabled, list_accounts
    from integrations.google.search import web_search_tools_enabled

    if google_tools_enabled() and list_accounts():
        return True
    if bookmarks_available():
        return True
    return web_search_tools_enabled()


def _overleaf_tools_wanted() -> bool:
    from integrations.overleaf.client import overleaf_tools_enabled

    return overleaf_tools_enabled()


def _shell_tools_wanted() -> bool:
    from integrations.shell.runner import shell_tools_enabled

    return shell_tools_enabled()


def _editor_tools_wanted() -> bool:
    from cli.code_assist import editor_tools_enabled

    return editor_tools_enabled()


def _memory_tools_wanted() -> bool:
    from processing.text.memory.tools import memory_tools_enabled

    return memory_tools_enabled()


def _skill_tools_wanted() -> bool:
    from processing.text.skills import skill_tools_enabled

    return skill_tools_enabled()


def _spawn_tools_in_schema(state: _SessionState) -> bool:
    from harness.subagent.tools import spawn_tools_in_schema

    return spawn_tools_in_schema(state)


def _tool_calls_openai_payload(tool_calls: list) -> list[dict[str, object]]:
    from cli.agent_runtime import tool_calls_openai_payload

    return tool_calls_openai_payload(tool_calls)


def _execute_speak_tool(state: _SessionState, args: dict) -> str:
    from processing.audio.speech.sources import tool_args_to_payload

    payload = tool_args_to_payload(args if isinstance(args, dict) else {})
    try:
        from processing.audio.speech.sources import resolve_speak_payload

        body, source = resolve_speak_payload(
            text=payload.get("text"),
            path=payload.get("path"),
            turn=payload.get("turn"),
            messages=state.messages,
            roots=_speak_roots(state),
        )
    except Exception as exc:
        return f"speak failed: {exc}"
    _emit(f"(speak {source}, {len(body)} chars)")
    try:
        _speak_now(state, body)
    except Exception as exc:
        return f"speak playback failed: {exc}"
    return f"spoken ({source}, {len(body)} chars)"


def _run_transcribe_request(state: _SessionState, path_token: str) -> str:
    from processing.audio.speech.listen import transcribe_path_blocking
    from processing.audio.speech.sources import resolve_workspace_file

    try:
        path = resolve_workspace_file(path_token, roots=_speak_roots(state))
    except Exception as exc:
        _emit(f"(sst failed: {exc})")
        return ""
    try:
        clip = _ensure_speech_store(state).ingest_file(path, role="user", source="file")
        _announce_speech_clip(clip)
    except Exception as exc:
        _emit(f"(speech clip save failed: {exc})")
    result = transcribe_path_blocking(
        engine=_sst_engine_singleton(state),
        path=path,
        language=state.sst_language,
        emit=_emit,
    )
    if not result.text:
        return ""
    lang = result.language or "auto"
    _emit(f"(sst file:{path}, {len(result.text)} chars, lang={lang})")
    _emit(result.text)
    return result.text


def transcribe_audio_paths(state: _SessionState, paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        text = _run_transcribe_request(state, str(path))
        if text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


def sst_recording(state: _SessionState) -> bool:
    mic = state.sst_mic
    if mic is None:
        return False
    active = getattr(mic, "active", False)
    finishing = getattr(mic, "finishing", False)
    return bool(active) and not bool(finishing)


def start_sst_recording(state: _SessionState, max_seconds: float | None = None) -> None:
    from processing.audio.speech.capture import MicRecorder, record_max_seconds_default

    if state.sst_busy:
        _emit("(sst busy: wait for the current transcript)")
        return
    if sst_recording(state):
        _emit("(sst already recording · /listen to stop)")
        return

    cap = record_max_seconds_default() if max_seconds is None else max(2.0, min(float(max_seconds), 300.0))

    def on_status(phase: str) -> None:
        _notify_mic(phase)

    recorder = MicRecorder(max_seconds=cap, on_status=on_status)
    state.sst_mic = recorder
    try:
        recorder.start()
    except Exception as exc:
        state.sst_mic = None
        _emit(f"(sst listen failed: {exc})")
        _notify_mic("idle")
        return
    _notify_mic("recording")
    _emit(
        f"(sst recording · Speak/Ctrl+L or /listen to stop · /listen-cancel to discard · max {cap:.0f}s)"
    )


def cancel_sst_recording(state: _SessionState) -> None:
    mic = state.sst_mic
    if mic is None:
        _emit("(sst: not recording)")
        return
    try:
        take = getattr(mic, "take_finish", None)
        if callable(take) and not take():
            return
        mic.cancel()
        mic.join(timeout=3.0)
    except Exception as exc:
        _emit(f"(sst cancel failed: {exc})")
    finally:
        state.sst_mic = None
        state.sst_busy = False
        _notify_mic("idle")
    _emit("(sst cancelled)")


def finish_sst_recording(state: _SessionState) -> str:
    from processing.audio.speech.capture import describe_capture, quiet_rms_threshold
    from processing.audio.speech.listen import transcribe_wave_blocking

    mic = state.sst_mic
    if mic is None:
        _emit("(sst: not recording)")
        return ""
    take = getattr(mic, "take_finish", None)
    if callable(take) and not take():
        return ""
    state.sst_busy = True
    _notify_mic("transcribing")
    try:
        mic.stop()
        mic.join(timeout=8.0)
        wave, rate, peak_rms = mic.result()
    except Exception as exc:
        state.sst_mic = None
        state.sst_busy = False
        _notify_mic("idle")
        _emit(f"(sst listen failed: {exc})")
        return ""

    stats = describe_capture(wave, rate, peak_rms)
    if getattr(mic, "hit_max", False):
        _emit(f"(sst hit max {getattr(mic, 'max_seconds', 0):.0f}s · captured {stats})")
    else:
        _emit(f"(sst captured {stats})")
    if peak_rms < quiet_rms_threshold():
        _emit("(sst warning: mic level is near silence. Check SOPHON_SST_MIC.)")

    try:
        clip = _ensure_speech_store(state).save_wave(wave, rate, role="user", source="mic")
        _announce_speech_clip(clip)
    except Exception as exc:
        _emit(f"(speech clip save failed: {exc})")

    try:
        _emit("(sst transcribing)")
        result = transcribe_wave_blocking(
            engine=_sst_engine_singleton(state),
            wave=wave,
            sample_rate=rate,
            language=state.sst_language,
            emit=_emit,
            peak_rms=peak_rms,
        )
        text = result.text.strip()
        if not text:
            _emit("(sst: no transcript produced. Nothing was inserted into the prompt.)")
            return ""
        lang = result.language or "auto"
        _emit(f"(sst listen, {len(text)} chars, lang={lang})")
        _dictate(text)
        return text
    finally:
        state.sst_mic = None
        state.sst_busy = False
        _notify_mic("idle")


def listen_from_microphone(state: _SessionState, max_seconds: float | None = None) -> str:
    if sst_recording(state):
        return finish_sst_recording(state)
    start_sst_recording(state, max_seconds=max_seconds)
    return ""


def _execute_transcribe_tool(state: _SessionState, args: dict) -> str:
    path = str(args.get("path") or "").strip()
    if not path:
        return "transcribe failed: missing path"
    text = _run_transcribe_request(state, path)
    if not text:
        return "transcribe failed: empty transcript"
    return f"transcript ({len(text)} chars):\n{text}"


def _tool_target_path(state: _SessionState, name: str, args: dict) -> Path | None:
    if name == "shell_cd":
        from integrations.shell.runner import ShellSession

        session = state.shell_session
        path = args.get("path")
        if path is None or str(path).strip() == "":
            return Path.home()
        if isinstance(session, ShellSession):
            return session.resolve_path(str(path))
        return Path(str(path)).expanduser()
    if name == "editor_propose_edit":
        from cli.code_assist import resolve_assist_path

        try:
            return resolve_assist_path(state, str(args.get("path") or ""))
        except (OSError, ValueError):
            return None
    if name == "memory_propose":
        layer = state.memory_layer
        markdown = getattr(layer, "markdown", None)
        facts = getattr(markdown, "facts_path", None)
        if facts is not None:
            return Path(facts)
    return None


def _authorize_tool(state: _SessionState, name: str, args: dict) -> str | None:
    from harness import ensure_harness

    harness = ensure_harness(state)
    cwd = Path(state.project_root)
    session = state.shell_session
    session_cwd = getattr(session, "cwd", None)
    if session_cwd is not None:
        cwd = Path(session_cwd)
    target = _tool_target_path(state, name, args)
    ask = None if int(getattr(state, "delegation_depth", 0) or 0) > 0 else _ask_permission
    result = harness.authorize(name, args, cwd=cwd, target=target, ask=ask)
    state.last_permission = result.permission
    if result.allowed:
        return None
    return result.message or "error: permission denied"


def _execute_one_tool(state: _SessionState, call) -> str:
    from cli.assist_tools import EDITOR_TOOL_NAMES, execute_editor_tool
    from integrations.google.tools import GOOGLE_TOOL_NAMES, execute_google_tool
    from integrations.obsidian.tools import VAULT_TOOL_NAMES, execute_vault_tool
    from integrations.overleaf.tools import OVERLEAF_TOOL_NAMES, execute_overleaf_tool
    from integrations.zotero.tools import ZOTERO_TOOL_NAMES, execute_zotero_tool
    from integrations.shell.runner import ShellSession
    from integrations.shell.tools import SHELL_TOOL_NAMES, execute_shell_tool
    from processing.audio.speech.tools import SPEAK_TOOL_NAME, TRANSCRIBE_TOOL_NAME
    from processing.text.memory.tools import (
        MEMORY_PROPOSE,
        MEMORY_TOOL_NAMES,
        execute_memory_tool,
    )
    from processing.text.skills import SKILL_TOOL_NAMES, execute_skill_tool
    from harness.subagent.tools import SUBAGENT_TOOL_NAMES, execute_subagent_tool

    name = getattr(call, "name", None) or ""
    args = getattr(call, "arguments", None)
    if not isinstance(args, dict):
        args = {}
    quiet = bool(getattr(state, "quiet", False))

    def progress(tool_name: str, preview: object) -> None:
        if quiet:
            return
        _tool_progress_line(tool_name, preview)

    if name in SHELL_TOOL_NAMES and state.shell_session is None:
        state.shell_session = ShellSession(cwd=Path(state.project_root).resolve())
    denied = _authorize_tool(state, str(name), args)
    if denied:
        return denied
    if name in SUBAGENT_TOOL_NAMES:
        preview = args.get("description") or args.get("prompt") or ""
        progress(name, preview)
        try:
            return execute_subagent_tool(state, str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    if name in MEMORY_TOOL_NAMES:
        preview = args.get("query") or args.get("text") or args.get("id") or args.get("ids") or ""
        progress(name, preview)
        try:
            if name == MEMORY_PROPOSE:
                raw_ids = args.get("ids")
                if isinstance(raw_ids, list):
                    ids = [str(i).strip() for i in raw_ids if str(i).strip()]
                else:
                    ids = str(raw_ids or "").split()
                return _queue_memory_promotion(state, ids)
            return execute_memory_tool(state.memory_layer, str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    if name in SKILL_TOOL_NAMES:
        preview = args.get("name") or args.get("path") or ""
        progress(name, preview)
        try:
            return execute_skill_tool(state.skill_catalog, str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    if name == SPEAK_TOOL_NAME:
        return _execute_speak_tool(state, args)
    if name == TRANSCRIBE_TOOL_NAME:
        return _execute_transcribe_tool(state, args)
    if name in VAULT_TOOL_NAMES:
        progress(name, args)
        try:
            return execute_vault_tool(str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    if name in ZOTERO_TOOL_NAMES:
        preview = args.get("query") or args.get("collection") or args.get("key") or ""
        progress(name, preview)
        try:
            return execute_zotero_tool(str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    if name in GOOGLE_TOOL_NAMES:
        preview = args.get("query") or args.get("id") or args.get("parent_id") or ""
        progress(name, preview)
        try:
            return execute_google_tool(str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    if name in OVERLEAF_TOOL_NAMES:
        preview = args.get("project_id") or args.get("path") or ""
        progress(name, preview)
        try:
            return execute_overleaf_tool(str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    if name in SHELL_TOOL_NAMES:
        if state.shell_session is None:
            state.shell_session = ShellSession(cwd=Path(state.project_root).resolve())
        session = state.shell_session
        assert isinstance(session, ShellSession)
        preview = args.get("command") or args.get("path") or ""
        progress(name, preview)
        try:
            return execute_shell_tool(session, str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    if name in EDITOR_TOOL_NAMES:
        preview = args.get("path") or args.get("description") or ""
        progress(name, preview)
        try:
            return execute_editor_tool(state, str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    return f"error: unknown tool {name!r}"


def _estimate_token_count(text: str) -> int:
    from cli.agent_runtime import estimate_token_count

    return estimate_token_count(text)


def _completion_token_counts(completion: object, *, text: str) -> tuple[int, int, str]:
    from cli.agent_runtime import completion_token_counts

    return completion_token_counts(completion, text=text)


def _session_tool_bundle(state: _SessionState) -> tuple[list[dict], str | None]:
    from cli.chat_tools import chat_tools_system_hint, default_chat_tools
    from cli.code_assist import editor_context_hint, ensure_assist
    from harness import ensure_harness, harness_system_hint

    ensure_assist(state)
    tts_on = _tts_tools_enabled()
    sst_on = _sst_tools_enabled()
    obs_on = _obsidian_tools_wanted()
    zot_on = _zotero_tools_wanted()
    google_on = _google_tools_wanted()
    overleaf_on = _overleaf_tools_wanted()
    shell_on = _shell_tools_wanted()
    editor_on = _editor_tools_wanted()
    memory_on = state.memory_layer is not None and _memory_tools_wanted()
    skill_on = state.skill_catalog is not None and _skill_tools_wanted()
    spawn_on = _spawn_tools_in_schema(state)
    tools = default_chat_tools(
        tts_tool=tts_on,
        sst_tool=sst_on,
        obsidian_tool=obs_on,
        zotero_tool=zot_on,
        google_tool=google_on,
        overleaf_tool=overleaf_on,
        shell_tool=shell_on,
        editor_tool=editor_on,
        memory_tool=memory_on,
        skill_tool=skill_on,
        subagent_tool=spawn_on,
    )
    hint = chat_tools_system_hint(
        tts_tool=tts_on,
        sst_tool=sst_on,
        obsidian_tool=obs_on,
        zotero_tool=zot_on,
        google_tool=google_on,
        overleaf_tool=overleaf_on,
        shell_tool=shell_on,
        editor_tool=editor_on,
        memory_tool=memory_on,
        skill_tool=skill_on,
        subagent_tool=spawn_on,
    )
    runtime = editor_context_hint(state) if editor_on else None
    extra = "\n\n".join(
        part
        for part in (
            hint,
            harness_system_hint(ensure_harness(state)),
            runtime,
        )
        if part
    )
    return tools, extra or None


def _inject_system_extra(call_messages: list[dict[str, object]], extra: str) -> list[dict[str, object]]:
    from cli.agent_runtime import inject_system_extra

    return inject_system_extra(call_messages, extra)


def _complete_chat_turn(
    state: _SessionState,
    messages: list[dict[str, object]],
    tools: list[dict] | None,
):
    from cli.agent_runtime import complete_chat_turn

    return complete_chat_turn(state, messages, tools)


def _run_tool_loop(
    state: _SessionState,
    call_messages: list[dict[str, object]],
    *,
    tools: list[dict],
) -> _ServerLoopOutcome:
    from cli.agent_runtime import run_tool_loop

    outcome = run_tool_loop(
        state,
        call_messages,
        tools=tools,
        execute_tool=_execute_one_tool,
        emit_think=_emit_think,
        emit_tool=_emit_tool_heartbeat,
        emit_line=_emit,
    )
    return _ServerLoopOutcome(
        text=outcome.text,
        elapsed_s=outcome.elapsed_s,
        spoke=outcome.spoke,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        stop_reason=outcome.stop_reason,
        reasoning=outcome.reasoning,
        tools=outcome.tools,
        rounds=outcome.rounds,
    )


def _run_generation(state: _SessionState) -> None:
    if not chat_session_ready(state):
        if is_server_backend(state.backend_id):
            _emit(
                f"No {state.backend_id} model selected. Use /models then /model NAME "
                f"(or start the server)."
            )
        else:
            _emit(
                "No model weights are loaded. Use /models to list presets, /model-download PRESET to fetch one, "
                "or /model PATH for a local directory with config.json. "
                "Or /backend lmstudio if LM Studio is running."
            )
        if state.messages and state.messages[-1].get("role") == "user":
            state.messages.pop()
        return

    cancelled = {"flag": False}
    prev_handler = signal.getsignal(signal.SIGINT)

    def _on_sigint(_signum: int, _frame: object) -> None:
        cancelled["flag"] = True
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        pass

    retrieval = _run_retrieval(state)
    call_messages, built = _build_call_messages(state, retrieval)
    call_messages = _inject_token_budget(call_messages, state.params.max_new_tokens)
    if state.debug and built.injected_blocks:
        _emit(f"(context: injected {', '.join(built.injected_blocks)})")
    _emit_context_heartbeat(state, built)

    tools, extra = _session_tool_bundle(state)
    if extra:
        call_messages = _inject_system_extra(call_messages, extra)

    try:
        loop_tools: list[ToolCallTrace] = []
        loop_rounds = 0
        reasoning: str | None = None
        plan_text: str | None = None
        if tools:
            outcome = _run_tool_loop(
                state,
                call_messages,
                tools=tools,
            )
            text = outcome.text
            elapsed = outcome.elapsed_s
            spoke = outcome.spoke
            prompt_tokens = outcome.prompt_tokens
            completion_tokens = outcome.completion_tokens
            stop_reason = outcome.stop_reason
            reasoning = outcome.reasoning
            loop_tools = outcome.tools
            loop_rounds = outcome.rounds
        elif is_server_backend(state.backend_id):
            from backend.chat_resolve import server_chat_complete

            t0 = time.perf_counter()
            completion = server_chat_complete(
                state.backend_id,
                model=str(state.server_model or ""),
                messages=_messages_for_server(call_messages),
                max_new_tokens=state.params.max_new_tokens,
                temperature=state.params.temperature,
                top_p=state.params.top_p,
                tools=None,
            )
            text = getattr(completion, "text", None)
            if text is None:
                text = str(completion)
            elapsed = time.perf_counter() - t0
            spoke = False
            prompt_tokens, completion_tokens, stop_reason = _completion_token_counts(
                completion,
                text=str(text),
            )
            reasoning = _completion_reasoning(completion)
            if reasoning:
                _emit_think(reasoning)
        else:
            meta = state.meta
            assert meta is not None
            result: GenerationResult = generate_response(
                state.processor,
                state.model,
                call_messages,
                max_new_tokens=state.params.max_new_tokens,
                enable_thinking=state.enable_thinking,
                temperature=state.params.temperature,
                top_p=state.params.top_p,
                top_k=state.params.top_k,
                repetition_penalty=state.params.repetition_penalty,
                seed=state.params.seed,
                strip=state.strip,
                extra_specials=meta.special_tokens,
                eos_token_ids=meta.eos_token_ids or None,
            )
            thought = getattr(result, "reasoning", None)
            if thought:
                _emit_think(str(thought))
            text = parsed_to_display_text(result.parsed)
            elapsed = result.gen_time_s
            spoke = False
            prompt_tokens = result.input_tokens
            completion_tokens = result.new_tokens
            stop_reason = result.stop_reason
            reasoning = getattr(result, "reasoning", None)
            plan_text = getattr(result, "plan", None)
    except KeyboardInterrupt:
        state.stats.record_exception()
        if state.messages and state.messages[-1].get("role") == "user":
            state.messages.pop()
        _emit("(generation cancelled)")
        return
    except Exception as exc:
        state.stats.record_exception()
        if state.messages and state.messages[-1].get("role") == "user":
            state.messages.pop()
        _emit(f"Generation failed: {exc}")
        return
    finally:
        try:
            signal.signal(signal.SIGINT, prev_handler)
        except (ValueError, OSError):
            pass
        _emit_heartbeat_end()

    _commit_assistant_turn(
        state,
        str(text),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed=elapsed,
        stop_reason=stop_reason,
        retrieval=retrieval,
        spoke=spoke,
        reasoning=reasoning,
        tools=loop_tools,
        tool_rounds=loop_rounds,
        plan_text=plan_text,
    )
    warn = state.stats.warn_if_context_high(0.9)
    if warn:
        _emit(warn)


def _init_rag_retrievers(params: ChatCliParams) -> tuple[RagRetriever | None, RagRetriever | None]:
    retriever: RagRetriever | None = None
    structure: RagRetriever | None = None
    rag_id = str(params.rag or "noop").strip().lower()
    try:
        candidate = load_rag_retriever(
            rag_id,
            native_index_path=params.rag_index,
            structure_dir=params.rag_structure_dir,
        )
        if candidate.backend_id() != "noop":
            retriever = candidate
    except Exception as exc:
        _emit(f"(retrieval init failed: {exc})")
        retriever = None
    structure_id = str(getattr(params, "rag_structure", "none") or "none").strip().lower()
    if structure_id not in ("", "none", "noop"):
        try:
            structure = load_rag_retriever(
                structure_id,
                structure_dir=params.rag_structure_dir,
            )
            if structure.backend_id() == "noop":
                structure = None
        except Exception as exc:
            _emit(f"(structure retrieval init failed: {exc})")
            structure = None
    return retriever, structure


def prepare_shell_chat_session(
    params: ChatCliParams,
    resolved_model_dir: Path,
    preset_key: str | None,
    *,
    backend_id: ChatBackendId = "hf",
    server_model: str | None = None,
) -> _SessionState:
    _suppress_noisy_warnings()

    enable_thinking = _thinking_default(params.thinking)
    root = sophon_project_root()
    quantization = resolve_cli_quantization(qbit=params.qbit, quantization=params.quantization)
    if is_server_backend(backend_id) and server_model:
        model_path_str = f"{backend_id}:{server_model}"
    else:
        model_path_str = str(resolved_model_dir.resolve())

    debug = initial_debug_mode_from_env() if params.debug is None else bool(params.debug)
    gen_params = _GenParams(
        max_new_tokens=int(params.max_new_tokens),
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        repetition_penalty=params.repetition_penalty,
        seed=params.seed,
    )
    stats = SessionStats(max_position_embeddings=None)

    retriever, structure_retriever = _init_rag_retrievers(params)

    memory_store = open_memory_store(params.memory_db)
    memory_scope: MemoryScope | None = None
    if memory_store is not None:
        session_id = params.memory_session or _dt.datetime.now().strftime("session_%Y%m%d_%H%M%S")
        memory_scope = MemoryScope(session_id=session_id, user_id=params.memory_user)
    memory_layer = open_memory_layer(memory_store, memory_scope, root)
    memory_budget = budget_from_env(max(int(params.memory_recall_turns), 0))

    tts_opts = params.tts
    tts_instruct = tts_opts.tts_instruct
    if isinstance(tts_instruct, str) and tts_instruct.strip() == "":
        tts_instruct = None
    tts_device = (
        tts_opts.tts_device.strip()
        if isinstance(tts_opts.tts_device, str) and tts_opts.tts_device.strip()
        else None
    )
    om_arg = tts_opts.tts_ollama_model
    ollama_model_resolved = om_arg.strip() if isinstance(om_arg, str) and om_arg.strip() else None
    sst_opts = params.sst
    sst_device = (
        sst_opts.sst_device.strip()
        if isinstance(sst_opts.sst_device, str) and sst_opts.sst_device.strip()
        else None
    )

    from harness import load_harness
    from integrations.shell.runner import ShellSession

    return _SessionState(
        processor=None,
        model=None,
        meta=None,
        messages=_baseline_messages(params.system),
        system_text=params.system,
        enable_thinking=enable_thinking,
        strip=not bool(params.raw),
        debug=debug,
        stats=stats,
        params=gen_params,
        retriever=retriever,
        structure_retriever=structure_retriever,
        retrieval_top_k=max(int(params.rag_top_k), 1),
        rag_adaptive=bool(getattr(params, "rag_adaptive", True)),
        memory=memory_store,
        memory_scope=memory_scope,
        memory_recall_turns=max(int(params.memory_recall_turns), 0),
        memory_layer=memory_layer,
        memory_budget=memory_budget,
        skill_catalog=open_skill_catalog(root),
        tts_enabled=bool(tts_opts.tts_enabled),
        tts_plain_text=not bool(tts_opts.tts_raw_output),
        tts_model_id=str(tts_opts.tts_model),
        tts_speaker=str(tts_opts.tts_speaker),
        tts_language=str(tts_opts.tts_language),
        tts_instruct=tts_instruct if isinstance(tts_instruct, str) else None,
        tts_max_chars=max(int(tts_opts.tts_max_chars), 1),
        tts_device=tts_device,
        tts_backend_id=tts_opts.tts_backend,
        tts_ollama_model=ollama_model_resolved,
        tts_speed=max(0.5, min(float(getattr(tts_opts, "tts_speed", 1.0) or 1.0), 2.0)),
        sst_enabled=bool(sst_opts.sst_enabled),
        sst_backend_id=sst_opts.sst_backend,
        sst_model_id=str(sst_opts.sst_model),
        sst_language=sst_opts.sst_language,
        sst_device=sst_device,
        sst_max_new_tokens=max(int(sst_opts.sst_max_new_tokens), 1),
        model_path=model_path_str,
        preset_key=None if is_server_backend(backend_id) else preset_key,
        quantization=quantization,
        project_root=root,
        backend_id=backend_id,
        server_model=server_model,
        shell_session=ShellSession(cwd=root.resolve()),
        tool_max_rounds=_tool_max_rounds_from_env(),
        speech_store=_make_speech_store(memory_scope),
        harness=load_harness(root),
    )


def prepare_chat_session(
    params: ChatCliParams,
    *,
    on_load_progress: Callable[[int, int, str], None] | None = None,
) -> _SessionState:
    _suppress_noisy_warnings()

    enable_thinking = _thinking_default(params.thinking)
    root = sophon_project_root()
    preset_key = params.preset
    if params.preset is not None:
        resolved_model_dir = resolve_preset_dir(params.preset, root)
    else:
        resolved_model_dir = resolve_local_model_dir(params.model)
    model_path = require_model_on_disk(resolved_model_dir)
    quantization = resolve_cli_quantization(qbit=params.qbit, quantization=params.quantization)

    processor, model = load_processor_and_model(
        model_path,
        quantization,
        on_load_progress=on_load_progress,
    )
    meta = read_model_meta(model_path, processor)

    debug = initial_debug_mode_from_env() if params.debug is None else bool(params.debug)
    gen_params = _GenParams(
        max_new_tokens=int(params.max_new_tokens),
        temperature=params.temperature if params.temperature is not None else meta.default_temperature,
        top_p=params.top_p if params.top_p is not None else meta.default_top_p,
        top_k=params.top_k if params.top_k is not None else meta.default_top_k,
        repetition_penalty=params.repetition_penalty,
        seed=params.seed,
    )
    stats = SessionStats(max_position_embeddings=meta.max_position_embeddings)

    retriever, structure_retriever = _init_rag_retrievers(params)

    memory_store = open_memory_store(params.memory_db)
    memory_scope: MemoryScope | None = None
    if memory_store is not None:
        session_id = params.memory_session or _dt.datetime.now().strftime("session_%Y%m%d_%H%M%S")
        memory_scope = MemoryScope(session_id=session_id, user_id=params.memory_user)
    memory_layer = open_memory_layer(memory_store, memory_scope, root)
    memory_budget = budget_from_env(max(int(params.memory_recall_turns), 0))

    tts_opts = params.tts
    tts_instruct = tts_opts.tts_instruct
    if isinstance(tts_instruct, str) and tts_instruct.strip() == "":
        tts_instruct = None
    tts_device = (
        tts_opts.tts_device.strip()
        if isinstance(tts_opts.tts_device, str) and tts_opts.tts_device.strip()
        else None
    )
    om_arg = tts_opts.tts_ollama_model
    ollama_model_resolved = om_arg.strip() if isinstance(om_arg, str) and om_arg.strip() else None
    sst_opts = params.sst
    sst_device = (
        sst_opts.sst_device.strip()
        if isinstance(sst_opts.sst_device, str) and sst_opts.sst_device.strip()
        else None
    )

    from harness import load_harness
    from integrations.shell.runner import ShellSession

    return _SessionState(
        processor=processor,
        model=model,
        meta=meta,
        messages=_baseline_messages(params.system),
        system_text=params.system,
        enable_thinking=enable_thinking,
        strip=not bool(params.raw),
        debug=debug,
        stats=stats,
        params=gen_params,
        retriever=retriever,
        structure_retriever=structure_retriever,
        retrieval_top_k=max(int(params.rag_top_k), 1),
        rag_adaptive=bool(getattr(params, "rag_adaptive", True)),
        memory=memory_store,
        memory_scope=memory_scope,
        memory_recall_turns=max(int(params.memory_recall_turns), 0),
        memory_layer=memory_layer,
        memory_budget=memory_budget,
        skill_catalog=open_skill_catalog(root),
        tts_enabled=bool(tts_opts.tts_enabled),
        tts_plain_text=not bool(tts_opts.tts_raw_output),
        tts_model_id=str(tts_opts.tts_model),
        tts_speaker=str(tts_opts.tts_speaker),
        tts_language=str(tts_opts.tts_language),
        tts_instruct=tts_instruct if isinstance(tts_instruct, str) else None,
        tts_max_chars=max(int(tts_opts.tts_max_chars), 1),
        tts_device=tts_device,
        tts_backend_id=tts_opts.tts_backend,
        tts_ollama_model=ollama_model_resolved,
        tts_speed=max(0.5, min(float(getattr(tts_opts, "tts_speed", 1.0) or 1.0), 2.0)),
        sst_enabled=bool(sst_opts.sst_enabled),
        sst_backend_id=sst_opts.sst_backend,
        sst_model_id=str(sst_opts.sst_model),
        sst_language=sst_opts.sst_language,
        sst_device=sst_device,
        sst_max_new_tokens=max(int(sst_opts.sst_max_new_tokens), 1),
        model_path=model_path,
        preset_key=preset_key,
        quantization=quantization,
        project_root=root,
        backend_id="hf",
        server_model=None,
        shell_session=ShellSession(cwd=root.resolve()),
        tool_max_rounds=_tool_max_rounds_from_env(),
        speech_store=_make_speech_store(memory_scope),
        harness=load_harness(root),
    )


def should_eager_load_at_startup(
    params: ChatCliParams,
    resolved_model_dir: Path,
    *,
    has_weights: bool,
) -> bool:
    if not has_weights:
        return False
    if params.preset is not None or params.model is not None:
        return True
    if params.startup_preload:
        return True
    env = os.environ.get("SOPHON_CHAT_PRELOAD", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    from utils.device.platform import is_wsl, is_windows_mount_path

    if is_wsl() and is_windows_mount_path(resolved_model_dir):
        return False
    if sys.platform == "win32":
        return True
    return False


def prepare_chat_session_or_shell(
    params: ChatCliParams,
    *,
    on_load_progress: Callable[[int, int, str], None] | None = None,
) -> _SessionState:
    root = sophon_project_root()
    try:
        backend_id = resolve_chat_backend()
    except ValueError as exc:
        _emit(f"(backend resolve failed: {exc}; using hf)")
        backend_id = "hf"

    if is_server_backend(backend_id):
        server_model = _pick_default_server_model(backend_id)
        return prepare_shell_chat_session(
            params,
            root,
            None,
            backend_id=backend_id,
            server_model=server_model,
        )

    preset_key, resolved_model_dir = resolve_chat_startup_model(
        preset=params.preset,
        model=params.model,
        cwd=root,
    )
    has_weights = (resolved_model_dir / "config.json").is_file() and model_dir_has_complete_weights(
        resolved_model_dir
    )
    if has_weights and should_eager_load_at_startup(params, resolved_model_dir, has_weights=True):
        load_params = params
        if params.preset is None and preset_key is not None:
            load_params = replace(params, preset=preset_key)
        return prepare_chat_session(load_params, on_load_progress=on_load_progress)
    return prepare_shell_chat_session(params, resolved_model_dir, preset_key, backend_id="hf")


def chat_startup_lines(state: _SessionState) -> list[str]:
    from cli.chat_display import chat_session_header_lines

    return [line.plain for line in chat_session_header_lines(state)]


def _episode_summary_from_messages(messages: list[dict[str, object]]) -> tuple[str, int]:
    user_turns = [str(m.get("content", "")).strip() for m in messages if m.get("role") == "user"]
    assistant_turns = [str(m.get("content", "")).strip() for m in messages if m.get("role") == "assistant"]
    turn_count = len(user_turns)
    if turn_count == 0:
        return "", 0
    first = user_turns[0][:200]
    last = user_turns[-1][:200] if turn_count > 1 else ""
    tail = assistant_turns[-1][:200] if assistant_turns else ""
    parts = [f"{turn_count} user turn(s). first: {first}"]
    if last:
        parts.append(f"last: {last}")
    if tail:
        parts.append(f"reply: {tail}")
    return " | ".join(parts), turn_count


def close_chat_session(state: _SessionState) -> None:
    if state.sst_mic is not None:
        try:
            cancel_sst_recording(state)
        except Exception:
            pass
    _unload_model_weights(state)
    if state.memory_layer is not None:
        summary, turn_count = _episode_summary_from_messages(state.messages)
        if turn_count > 0:
            state.memory_layer.record_episode_on_close(summary, turn_count)
    if state.memory is not None:
        try:
            state.memory.close()
        except Exception:
            pass


dispatch_chat_line = _dispatch_command
run_chat_generation = _run_generation


def run_chat(params: ChatCliParams) -> None:
    configure_chat_io(_DEFAULT_IO)
    state: _SessionState | None = None
    try:
        try:
            state = prepare_chat_session_or_shell(params)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc

        for line in chat_startup_lines(state):
            _emit(line)

        last_sigint_ts: float = 0.0
        while not state.exit_requested:
            try:
                raw_line = _read_user_input("> ")
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - last_sigint_ts < 2.0:
                    _emit("(received Ctrl-C twice, exiting)")
                    break
                last_sigint_ts = now
                _emit("(use /quit or q to exit, or Ctrl-C again within 2s)")
                continue
            if raw_line is None:
                print(file=sys.stderr)
                break
            if raw_line.strip() == "":
                if sst_recording(state):
                    finish_sst_recording(state)
                continue
            handled, should_generate = dispatch_chat_line(state, raw_line)
            if state.exit_requested:
                break
            if handled and not should_generate:
                continue
            if not handled:
                state.messages.append(
                    {"role": "user", "content": prepare_user_message_text(state, raw_line)}
                )
            run_chat_generation(state)
    finally:
        if state is not None:
            close_chat_session(state)
        reset_chat_io()


def main() -> None:
    import sys as _sys

    from cli.terminal import chat_command as _chat

    _chat.main(args=_sys.argv[1:], prog_name="sophon-chat-cli", standalone_mode=True)


if __name__ == "__main__":
    main()
