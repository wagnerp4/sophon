from __future__ import annotations

import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

# Todo: remove this path bootstrap after the package uses consistent orodruin.* imports end-to-end.

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
    chat_backend_ids,
    is_server_backend,
    list_server_models,
    lmstudio_reachable,
    normalize_chat_backend_token,
    ollama_reachable,
    resolve_chat_backend,
    server_chat_complete,
)

from backend.hf.paths import (
    infer_default_quantization,
    require_model_on_disk,
    resolve_cli_quantization,
    resolve_local_model_dir,
)
from processing.text.context import ContextBuildResult, build_messages_for_model
from processing.text.memory import MemoryScope, MemoryStore, open_memory_store
from processing.text.retrieval import RagRetriever, RetrievalQuery, RetrievalResult, load_rag_retriever, rag_retriever_ids
from processing.audio.speech.cli import SttCliOptions, TtsCliOptions
from processing.audio.speech.factory import create_speech_stt_engine, create_speech_tts_engine
from processing.audio.speech.protocols import SpeechSttEngine, SpeechTtsEngine
from processing.audio.speech.speak import speak_text_blocking
from processing.audio.speech.types import (
    TtsBackendId,
    SttBackendId,
    normalize_stt_backend_token,
    normalize_stt_language,
    normalize_tts_backend_token,
    stt_backend_help_tokens,
    tts_backend_help_tokens,
)
from utils.device.env_bootstrap import orodruin_chat_logs_dir, orodruin_project_root
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
    split_hub_progress_label,
    throttled_progress_callback,
)


@dataclass
class _GenParams:
    max_new_tokens: int = 512
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
    memory_db: str | None
    memory_session: str | None
    memory_user: str | None
    memory_recall_turns: int
    tts: TtsCliOptions
    sst: SttCliOptions
    chat_first: bool = False
    startup_preload: bool = False


@dataclass(frozen=True)
class ChatIo:
    emit: Callable[[str], None]
    on_assistant: Callable[[str], None]
    on_dictate: Callable[[str], None] | None = None


def _default_emit(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _default_on_assistant(text: str) -> None:
    print(text)
    print(flush=True)


_DEFAULT_IO = ChatIo(emit=_default_emit, on_assistant=_default_on_assistant)
_CURRENT_IO = _DEFAULT_IO


def configure_chat_io(io: ChatIo) -> None:
    global _CURRENT_IO
    _CURRENT_IO = io


def reset_chat_io() -> None:
    global _CURRENT_IO
    _CURRENT_IO = _DEFAULT_IO


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
    retrieval_top_k: int = 5
    memory: MemoryStore | None = None
    memory_scope: MemoryScope | None = None
    memory_recall_turns: int = 6
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
    model_path: str = ""
    preset_key: str | None = None
    quantization: str = "none"
    project_root: Path = field(default_factory=Path.cwd)
    model_switching: bool = False
    adapter_path: str | None = None
    last_finetune_run_dir: Path | None = None
    finetune_running: bool = False
    backend_id: ChatBackendId = "hf"
    server_model: str | None = None
    last_retrieval_metrics: dict | None = None
    retrieval_probe_history: list = field(default_factory=list)
    shell_session: object | None = None
    tool_max_rounds: int | None = None


def _format_tool_max_rounds(value: int | None) -> str:
    if value is None:
        return "unlimited"
    return str(value)


def _tool_max_rounds_from_env() -> int | None:
    raw = os.environ.get("ORODRUIN_TOOL_MAX_ROUNDS", "").strip().lower()
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
        ("help", "quit", "reset", "save", "pop", "regen", "system", "tokens", "stats", "debug"),
    ),
    (
        "model",
        ("backend", "models", "model", "model-download"),
    ),
    (
        "generation",
        ("params", "temp", "top-p", "top-k", "max", "seed", "rep", "tool-rounds", "unlimited"),
    ),
    (
        "tools",
        ("tools", "obsidian-status", "mode"),
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
            "sst",
            "sst-backend",
            "sst-model",
            "sst-lang",
        ),
    ),
    (
        "memory",
        ("memory-status", "memory-clear", "rag-status", "rag-probe"),
    ),
    (
        "train",
        ("finetune", "finetune-status", "finetune-datasets", "adapter"),
    ),
)


def _register(name: str, help_text: str, shortcut: str | None = None):
    def deco(fn: Callable[["_SessionState", str], bool]) -> Callable[["_SessionState", str], bool]:
        _COMMANDS[name] = _Command(name=name, help_text=help_text, handler=fn)
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


def _format_help_command(cmd: _Command, shortcut_for: dict[str, str]) -> str:
    sc = shortcut_for.get(cmd.name)
    head = f"  /{cmd.name}" + (f" | {sc}" if sc else "")
    return f"{head:<24} {cmd.help_text}"


def _format_help(section: str | None = None) -> str:
    shortcut_for = {v: k for k, v in _SHORTCUTS.items()}
    listed: set[str] = set()
    blocks: list[tuple[str, list[str]]] = []
    for name, cmd_names in _HELP_SECTIONS:
        lines: list[str] = []
        for cmd_name in cmd_names:
            cmd = _COMMANDS.get(cmd_name)
            if cmd is None:
                continue
            listed.add(cmd_name)
            lines.append(_format_help_command(cmd, shortcut_for))
        if lines:
            blocks.append((name, lines))
    other_lines = [
        _format_help_command(cmd, shortcut_for)
        for cmd_name, cmd in _COMMANDS.items()
        if cmd_name not in listed
    ]
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
    rows.append("  /help [section]          Filter by section (session model generation tools speech memory train).")
    rows.append("  \"\"\"multi-line\"\"\"        Triple-quoted input is gathered until the closing \"\"\".")
    return "\n".join(rows)


format_chat_help = _format_help


@dataclass(frozen=True)
class ModelPickerEntry:
    target: str
    title: str
    detail: str
    local: bool
    current: bool
    kind: Literal["header", "preset", "path"] = "preset"


def chat_session_has_weights(state: _SessionState) -> bool:
    return state.processor is not None and state.model is not None and state.meta is not None


def chat_session_ready(state: _SessionState) -> bool:
    if is_server_backend(state.backend_id):
        return bool(state.server_model and str(state.server_model).strip())
    return chat_session_has_weights(state)


def _content_to_text(content: object) -> str:
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


def _messages_for_server(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        item: dict[str, object] = {"role": role}
        if role == "tool":
            item["content"] = _content_to_text(msg.get("content"))
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
                item["content"] = _content_to_text(content)
            item["tool_calls"] = msg["tool_calls"]
        else:
            item["content"] = _content_to_text(msg.get("content"))
        out.append(item)
    return out


def _pick_default_server_model(backend_id: ChatBackendId) -> str | None:
    try:
        names = list_server_models(backend_id)
    except Exception:
        return None
    return names[0] if names else None


def _parse_server_target(token: str) -> tuple[ChatBackendId, str] | None:
    raw = token.strip()
    lower = raw.lower()
    for prefix, backend in (("ollama:", "ollama"), ("lmstudio:", "lmstudio"), ("lms:", "lmstudio")):
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
) -> None:
    entries.append(ModelPickerEntry("", backend, "", False, False, "header"))
    if not reachable:
        entries.append(
            ModelPickerEntry("", f"({backend} unreachable)", "", False, False, "header")
        )
        return
    try:
        names = list_server_models(backend)
    except Exception as exc:
        entries.append(
            ModelPickerEntry("", f"(list failed: {exc})", "", False, False, "header")
        )
        return
    if not names:
        entries.append(
            ModelPickerEntry("", "(no models exposed)", "", False, False, "header")
        )
        return
    for name in names:
        entries.append(
            ModelPickerEntry(
                target=f"{backend}:{name}",
                title=name,
                detail=backend,
                local=True,
                current=_is_current_server_entry(state, backend, name),
                kind="preset",
            )
        )


def switch_to_server_model(state: _SessionState, backend: ChatBackendId, name: str) -> None:
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
        _emit(f"usage: /backend [{' | '.join(chat_backend_ids())}]")
        return
    if backend == state.backend_id:
        _emit(f"(already on backend {backend})")
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
        entries.append(ModelPickerEntry("", "huggingface (local)", "", False, False, "header"))
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
                )
            )

    if missing:
        entries.append(ModelPickerEntry("", "huggingface (not downloaded)", "", False, False, "header"))
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
                )
            )
    return entries


def build_model_picker_entries(state: _SessionState) -> list[ModelPickerEntry]:
    entries: list[ModelPickerEntry] = []
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
        ("ollama:", "lmstudio:")
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
    if ":" in raw and not raw.lower().startswith(("ollama:", "lmstudio:", "lms:", "hf:")):
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
    if view_norm not in ("all", "local", "downloaded", "missing", "remote"):
        return "usage: /models [all | local | missing]"

    lines: list[str] = [
        f"current: {_current_model_label(state)}",
        "",
    ]

    def append_server_block(backend: ChatBackendId, reachable: bool) -> None:
        lines.append(f"{backend}:")
        if not reachable:
            lines.append(f"  (unreachable)")
            lines.append("")
            return
        try:
            names = list_server_models(backend)
        except Exception as exc:
            lines.append(f"  (list failed: {exc})")
            lines.append("")
            return
        if not names:
            lines.append("  (no models exposed)")
            lines.append("")
            return
        for name in names:
            marker = "*" if _is_current_server_entry(state, backend, name) else " "
            lines.append(_format_model_line(marker=marker, label=name, repo_or_path=f"{backend}:{name}"))
        lines.append("")

    if view_norm == "all":
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
        for backend in ("ollama", "lmstudio"):
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

    preset_key, model_dir = _resolve_model_target(raw, state.project_root)
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
    if is_server_backend(state.backend_id):
        return f"backend={state.backend_id} model={state.server_model or '(none)'}"
    if state.preset_key:
        preset = HF_MODEL_PRESETS[state.preset_key]
        return f"preset={state.preset_key} repo={preset.repo_id} path={state.model_path}"
    if state.model_path.strip():
        return f"path={state.model_path}"
    return "(no model directory; use /models or /model PATH)"


@_register("help", "Show this help. /help [section] filters session|model|generation|tools|speech|memory|train.", shortcut="?")
def _cmd_help(state: _SessionState, arg: str) -> bool:
    _ = state
    _emit(_format_help(arg))
    return False


@_register("backend", "Show or switch chat backend: /backend [auto|ollama|lmstudio|hf].")
def _cmd_backend(state: _SessionState, arg: str) -> bool:
    token = arg.strip()
    if not token:
        _emit(f"backend={state.backend_id}  (set ORODRUIN_CHAT_BACKEND=auto|ollama|lmstudio|hf)")
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


@_register("models", "List all sources: ollama, lmstudio, huggingface.")
def _cmd_models(state: _SessionState, arg: str) -> bool:
    _emit(format_model_catalog(state, view=arg.strip() or "all"))
    return False


@_register("model", "Show or switch model: /model [ollama:NAME | lmstudio:NAME | PRESET | PATH].")
def _cmd_model(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(_current_model_label(state))
        return False
    try:
        switch_session_model(state, arg)
    except ValueError as exc:
        _emit(str(exc))
    return False


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

        def on_download_progress(n: int, total: int, label: str) -> None:
            phase, _ = split_hub_progress_label(label)
            status = format_download_progress(n, total, phase)
            sys.stderr.write(f"\r[orodruin] {status.replace(chr(10), ' | ')}    ")
            sys.stderr.flush()

        tqdm_class = hub_tqdm_bridge_factory(throttled_progress_callback(on_download_progress))
        with capture_hub_download_logs(lambda text: _emit(text)):
            with capture_hub_user_warnings(lambda text: _emit(f"(warning: {text})")):
                path = download_preset_snapshot(
                    preset_key,
                    tqdm_class=tqdm_class,
                    verbose=True,
                    on_status=on_download_status,
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
    default_preset: str | None,
) -> tuple[str, str | None, list[str]]:
    tokens = arg.split()
    dataset_id = "gsm8k_instructions"
    preset_key = default_preset
    idx = 0
    if idx < len(tokens):
        from training.finetune.datasets.registry import match_dataset_preset

        matched = match_dataset_preset(tokens[idx])
        if matched is not None:
            dataset_id = matched
            idx += 1
    if idx < len(tokens) and tokens[idx].lower() == "on":
        idx += 1
        if idx < len(tokens):
            try:
                preset_key = match_preset_key(tokens[idx])
            except ValueError:
                preset_key = tokens[idx]
            idx += 1
    overrides = tokens[idx:]
    return dataset_id, preset_key, overrides


def _format_finetune_dataset_catalog() -> str:
    from training.finetune.datasets.registry import FINETUNE_DATASET_PRESETS, dataset_preset_keys_sorted

    lines = ["finetune datasets:"]
    for key in dataset_preset_keys_sorted():
        preset = FINETUNE_DATASET_PRESETS[key]
        lines.append(f"  {key}: hub={preset.hub_id} max_examples={preset.max_examples}")
        lines.append(f"    {preset.description}")
    lines.append("")
    lines.append("usage: /finetune [DATASET] [on PRESET] [KEY=VAL ...]")
    lines.append("  default DATASET=gsm8k_instructions; PRESET=current selection")
    lines.append("  overrides: max_examples=500 epochs=1 lr=2e-4 batch_size=2")
    return "\n".join(lines)


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


@_register("finetune-datasets", "List finetune dataset presets.")
def _cmd_finetune_datasets(state: _SessionState, _arg: str) -> bool:
    _ = state
    _emit(_format_finetune_dataset_catalog())
    return False


@_register("finetune", "LoRA finetune current preset: /finetune [DATASET] [on PRESET] [KEY=VAL ...].")
def _cmd_finetune(state: _SessionState, arg: str) -> bool:
    dataset_id, preset_key, overrides = _parse_finetune_command(arg, state.preset_key)
    if not preset_key:
        _emit("usage: /finetune [DATASET] on PRESET [KEY=VAL ...]  (or select a preset first)")
        return False

    from training.finetune.job import run_finetune_job

    _unload_model_weights(state)
    state.finetune_running = True
    try:
        _emit(f"starting finetune preset={preset_key!r} dataset={dataset_id!r} ...")

        def on_log(msg: str) -> None:
            _emit(msg)

        def on_progress(step: int, total: int, label: str) -> None:
            if total > 0:
                _emit(f"train: {label} ({step}/{total})")
            else:
                _emit(f"train: {label}")

        result = run_finetune_job(
            preset_key,
            dataset_id,
            project_root=state.project_root,
            backend_id="auto",
            recipe_override_tokens=overrides,
            on_progress=on_progress,
            on_log=on_log,
        )
    except Exception as exc:
        _emit(f"(finetune failed: {exc})")
        return False
    finally:
        state.finetune_running = False

    state.preset_key = preset_key
    state.last_finetune_run_dir = result.run_dir
    state.adapter_path = str(result.adapter_dir.resolve())
    _emit(f"(finetune done; adapter at {result.adapter_dir})")
    _emit(f"(backend={result.backend_id}; run /adapter load to chat with the adapter)")
    return False


@_register("finetune-status", "Show last finetune run summary.")
def _cmd_finetune_status(state: _SessionState, _arg: str) -> bool:
    run_dir = state.last_finetune_run_dir
    if run_dir is None:
        _emit("(no finetune run in this session)")
        return False
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        _emit(f"(run dir exists but no run_meta.json: {run_dir})")
        return False
    from training.common.types import RunMeta

    meta = RunMeta.read_json(meta_path)
    _emit(f"run_dir: {run_dir}")
    _emit(f"preset: {meta.preset_key}  dataset: {meta.dataset_id}  backend: {meta.backend_id}")
    if meta.train_metrics:
        _emit(f"train_metrics: {meta.train_metrics}")
    if state.adapter_path:
        _emit(f"adapter_path: {state.adapter_path}")
    return False


@_register("adapter", "LoRA adapter: /adapter show | load [PATH] | clear.")
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
        return False

    if action == "clear":
        state.adapter_path = None
        if state.preset_key or state.model_path:
            _reload_current_session_model(state)
        else:
            _emit("(adapter cleared)")
        return False

    if action == "load":
        if rest:
            candidate = Path(rest).expanduser()
            if not candidate.is_dir():
                _emit(f"adapter path not found: {candidate}")
                return False
            state.adapter_path = str(candidate.resolve())
        elif state.adapter_path:
            pass
        elif state.last_finetune_run_dir is not None:
            state.adapter_path = str(state.last_finetune_run_dir.resolve())
        else:
            _emit("usage: /adapter load [PATH]  (or run /finetune first)")
            return False
        if not state.preset_key and not state.model_path.strip():
            _emit(f"(adapter set to {state.adapter_path}; use /model PRESET to load)")
            return False
        _reload_current_session_model(state)
        return False

    _emit("usage: /adapter show | load [PATH] | clear")
    return False


@_register("quit", "Exit chat.", shortcut="q")
def _cmd_quit(state: _SessionState, _arg: str) -> bool:
    state.exit_requested = True
    return False


@_register("reset", "Clear conversation (keeps system prompt; memory DB untouched).")
def _cmd_reset(state: _SessionState, _arg: str) -> bool:
    state.messages = _baseline_messages(state.system_text)
    state._last_persisted_user_obj_id = 0
    _emit("(conversation cleared)")
    return False


@_register("save", "Save transcript to <path> or data/chat_logs/<timestamp>.txt", shortcut="s")
def _cmd_save(state: _SessionState, arg: str) -> bool:
    target = arg.strip()
    if not target:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        target = str(orodruin_chat_logs_dir() / f"{ts}.txt")
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
    if state.retriever is None:
        _emit("retrieval: disabled (--rag leann --rag-index PATH)")
        return False
    _emit(f"retrieval: backend={state.retriever.backend_id()} top_k={state.retrieval_top_k}")
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
    "Estimate LEANN query cost: /rag-probe [N] [query text]. Uses last user turn if query omitted.",
)
def _cmd_rag_probe(state: _SessionState, arg: str) -> bool:
    if state.retriever is None:
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
    _emit(f"(rag-probe n={repeats} top_k={state.retrieval_top_k} query_chars={len(query)})")
    rows: list[RetrievalQueryMetrics] = []
    for i in range(repeats):
        try:
            import time

            t0 = time.perf_counter()
            result = state.retriever.retrieve(
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
                    backend_id=str(raw.get("backend_id") or state.retriever.backend_id()),
                    extras=dict(raw.get("extras") or {}),
                )
            else:
                metrics = build_query_metrics(
                    latency_s=latency,
                    chunks=list(result.chunks),
                    top_k=state.retrieval_top_k,
                    query_chars=len(query),
                    backend_id=state.retriever.backend_id(),
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
    _emit(f"obsidian tools: {'on' if enabled else 'off'} (ORODRUIN_OBSIDIAN_TOOLS)")
    _emit(f"api url: {url}")
    _emit(f"api key: {'set' if key else 'missing'}")
    if not enabled:
        _emit("enable with ORODRUIN_OBSIDIAN_TOOLS=1 and restart chat")
        return False
    try:
        info = ObsidianClient(timeout_s=5.0).ping()
        _emit(f"reachable: yes ({info})")
    except Exception as exc:
        _emit(f"reachable: no ({exc})")
    return False


@_register("tools", "Show enabled chat tools (speak / vault / shell).")
@_register("mode", "Alias for /tools (tool packs are unified).")
def _cmd_tools(state: _SessionState, _arg: str) -> bool:
    from integrations.obsidian.client import obsidian_tools_enabled
    from integrations.shell.runner import ShellSession, shell_tools_enabled
    from utils.device.env_bootstrap import DOTENV_LOAD_PATH

    if state.shell_session is None:
        state.shell_session = ShellSession(cwd=Path(state.project_root).resolve())
    session = state.shell_session
    assert isinstance(session, ShellSession)

    tts_on = _tts_tools_enabled()
    sst_on = _sst_tools_enabled()
    obs_on = obsidian_tools_enabled()
    shell_on = shell_tools_enabled()
    _emit(f"dotenv: {DOTENV_LOAD_PATH if DOTENV_LOAD_PATH else '(not loaded)'}")
    _emit(f"speak tool: {'on' if tts_on else 'off'} (ORODRUIN_TTS_TOOL)")
    _emit(f"transcribe tool: {'on' if sst_on else 'off'} (ORODRUIN_SST_TOOL)")
    _emit(f"vault tools: {'on' if obs_on else 'off'} (ORODRUIN_OBSIDIAN_TOOLS)")
    _emit(f"shell tools: {'on' if shell_on else 'off'} (ORODRUIN_SHELL_TOOLS)")
    _emit(f"shell cwd: {session.cwd}")
    _emit(f"tool rounds: {_format_tool_max_rounds(state.tool_max_rounds)} (/tool-rounds, /unlimited)")
    if tts_on:
        _emit("speak: speak")
    if sst_on:
        _emit("sst: transcribe")
    if obs_on:
        _emit("vault: vault_search, vault_list, vault_read, vault_recent")
    if shell_on:
        _emit("shell: shell_pwd, shell_cd, shell_ls, shell_read, shell_exec")
    if not obs_on and not shell_on and not tts_on and not sst_on:
        _emit("no chat tools enabled; set env flags in .env and restart")
    else:
        _emit("all enabled tool packs are available together (no mode switch)")
    return False


@_register("memory-status", "Show memory store and session info.")
def _cmd_memory_status(state: _SessionState, _arg: str) -> bool:
    if state.memory is None or state.memory_scope is None:
        _emit("memory: disabled (pass --memory-db or set ORODRUIN_MEMORY_DB)")
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


@_register("listen", "Record from the mic then put transcript in the prompt: /listen [SECONDS]. TUI: Ctrl+L.")
def _cmd_listen(state: _SessionState, arg: str) -> bool:
    token = arg.strip()
    seconds: float | None = None
    if token:
        try:
            seconds = float(token)
        except ValueError:
            _emit("usage: /listen [SECONDS]")
            return False
        seconds = max(1.0, min(seconds, 120.0))
    text = listen_from_microphone(state, max_seconds=seconds)
    if text.strip():
        _dictate(text)
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
            f"in the server UI (not via orodruin per request). Reply budget is /max."
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


def _dispatch_command(state: _SessionState, line: str) -> tuple[bool, bool]:
    """Returns (handled, should_generate)."""
    if not line:
        return True, False
    stripped = line.strip()
    if stripped.lower() in ("exit", "quit"):
        state.exit_requested = True
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
        return True, cmd.handler(state, arg)

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
    if state.retriever is None:
        return None
    query_text = _latest_user_text(state)
    if not query_text:
        return None
    try:
        import time

        t0 = time.perf_counter()
        result = state.retriever.retrieve(
            RetrievalQuery(text=query_text, params={"top_k": state.retrieval_top_k}),
        )
        if not isinstance(result.extras.get("metrics"), dict):
            from processing.text.retrieval.metrics import build_query_metrics

            metrics = build_query_metrics(
                latency_s=time.perf_counter() - t0,
                chunks=list(result.chunks),
                top_k=state.retrieval_top_k,
                query_chars=len(query_text),
                backend_id=state.retriever.backend_id(),
            )
            result.extras["metrics"] = metrics.as_dict()
        _record_retrieval_metrics(state, result)
        return result
    except Exception as exc:
        _emit(f"(retrieval failed: {exc})")
        return None


def _load_memory_turns(state: _SessionState) -> list[object]:
    if state.memory is None or state.memory_scope is None or state.memory_recall_turns <= 0:
        return []
    try:
        return list(state.memory.load_recent_turns(state.memory_scope, state.memory_recall_turns))
    except Exception as exc:
        _emit(f"(memory load failed: {exc})")
        return []


def _build_call_messages(
    state: _SessionState,
    retrieval: RetrievalResult | None,
) -> tuple[list[dict[str, object]], ContextBuildResult]:
    memory_turns = _load_memory_turns(state)
    if retrieval is None and not memory_turns:
        return list(state.messages), ContextBuildResult(messages=list(state.messages))
    base = [m for m in state.messages if m.get("role") != "system"]
    built = build_messages_for_model(
        base,
        retrieval=retrieval,
        memory_turns=memory_turns,
        system_text=state.system_text,
        max_retrieval_chunks=state.retrieval_top_k,
    )
    return built.messages, built


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


def _speak_now(state: _SessionState, text: str) -> None:
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
    raw = os.environ.get("ORODRUIN_TTS_TOOL", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _sst_tools_enabled() -> bool:
    raw = os.environ.get("ORODRUIN_SST_TOOL", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _obsidian_tools_wanted() -> bool:
    from integrations.obsidian.client import obsidian_tools_enabled

    return obsidian_tools_enabled()


def _shell_tools_wanted() -> bool:
    from integrations.shell.runner import shell_tools_enabled

    return shell_tools_enabled()


def _tool_calls_openai_payload(tool_calls: list) -> list[dict[str, object]]:
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


def listen_from_microphone(state: _SessionState, max_seconds: float | None = None) -> str:
    from processing.audio.speech.listen import listen_and_transcribe_blocking

    result = listen_and_transcribe_blocking(
        engine=_sst_engine_singleton(state),
        language=state.sst_language,
        emit=_emit,
        max_seconds=max_seconds,
    )
    if not result.text.strip():
        return ""
    lang = result.language or "auto"
    _emit(f"(sst listen, {len(result.text)} chars, lang={lang})")
    return result.text


def _execute_transcribe_tool(state: _SessionState, args: dict) -> str:
    path = str(args.get("path") or "").strip()
    if not path:
        return "transcribe failed: missing path"
    text = _run_transcribe_request(state, path)
    if not text:
        return "transcribe failed: empty transcript"
    return f"transcript ({len(text)} chars):\n{text}"


def _execute_one_tool(state: _SessionState, call) -> str:
    from integrations.obsidian.tools import VAULT_TOOL_NAMES, execute_vault_tool
    from integrations.shell.runner import ShellSession
    from integrations.shell.tools import SHELL_TOOL_NAMES, execute_shell_tool
    from processing.audio.speech.tools import SPEAK_TOOL_NAME, TRANSCRIBE_TOOL_NAME

    name = getattr(call, "name", None) or ""
    args = getattr(call, "arguments", None)
    if not isinstance(args, dict):
        args = {}
    if name == SPEAK_TOOL_NAME:
        return _execute_speak_tool(state, args)
    if name == TRANSCRIBE_TOOL_NAME:
        return _execute_transcribe_tool(state, args)
    if name in VAULT_TOOL_NAMES:
        _emit(f"({name} {args})")
        try:
            return execute_vault_tool(str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    if name in SHELL_TOOL_NAMES:
        if state.shell_session is None:
            state.shell_session = ShellSession(cwd=Path(state.project_root).resolve())
        session = state.shell_session
        assert isinstance(session, ShellSession)
        preview = args.get("command") or args.get("path") or ""
        _emit(f"({name} {preview})")
        try:
            return execute_shell_tool(session, str(name), args)
        except Exception as exc:
            return f"error: {exc}"
    return f"error: unknown tool {name!r}"


def _estimate_token_count(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return max(1, (len(cleaned) + 3) // 4)


def _completion_token_counts(completion: object, *, text: str) -> tuple[int, int, str]:
    prompt_tokens = getattr(completion, "prompt_tokens", None)
    completion_tokens = getattr(completion, "completion_tokens", None)
    finish_reason = getattr(completion, "finish_reason", None)
    in_tok = int(prompt_tokens) if isinstance(prompt_tokens, int) else 0
    out_tok = int(completion_tokens) if isinstance(completion_tokens, int) else _estimate_token_count(text)
    stop = str(finish_reason) if finish_reason else "stop"
    return in_tok, out_tok, stop


def _run_server_tool_loop(
    state: _SessionState,
    call_messages: list[dict[str, object]],
    *,
    tools: list[dict],
) -> tuple[str, float, bool, int, int, str]:
    from backend.chat_resolve import server_chat_complete

    assert state.server_model is not None
    working = [dict(m) for m in call_messages]
    t0 = time.perf_counter()
    spoke = False
    text = ""
    prompt_tokens = 0
    completion_tokens = 0
    stop_reason = "stop"
    max_rounds = state.tool_max_rounds
    round_i = 0
    while True:
        completion = server_chat_complete(
            state.backend_id,
            model=state.server_model,
            messages=_messages_for_server(working),
            max_new_tokens=state.params.max_new_tokens,
            temperature=state.params.temperature,
            top_p=state.params.top_p,
            tools=tools,
        )
        text = str(getattr(completion, "text", None) or "")
        in_tok, out_tok, stop_reason = _completion_token_counts(completion, text=text)
        if in_tok > 0:
            prompt_tokens = in_tok
        completion_tokens += out_tok
        tool_calls = list(getattr(completion, "tool_calls", None) or [])
        if not tool_calls:
            break
        assistant_msg: dict[str, object] = {
            "role": "assistant",
            "content": text if text.strip() else None,
            "tool_calls": _tool_calls_openai_payload(tool_calls),
        }
        working.append(assistant_msg)
        for call in tool_calls:
            name = getattr(call, "name", "") or ""
            if name == "speak":
                spoke = True
            result = _execute_one_tool(state, call)
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
            _emit("(tool loop hit max rounds; requesting final answer)")
            completion = server_chat_complete(
                state.backend_id,
                model=state.server_model,
                messages=_messages_for_server(working),
                max_new_tokens=state.params.max_new_tokens,
                temperature=state.params.temperature,
                top_p=state.params.top_p,
                tools=None,
            )
            text = str(getattr(completion, "text", None) or text)
            in_tok, out_tok, stop_reason = _completion_token_counts(completion, text=text)
            if in_tok > 0:
                prompt_tokens = in_tok
            completion_tokens += out_tok
            break
    elapsed = time.perf_counter() - t0
    if not str(text).strip() and spoke:
        text = "(spoken via speak tool)"
    if completion_tokens <= 0:
        completion_tokens = _estimate_token_count(text)
    return text, elapsed, spoke, prompt_tokens, completion_tokens, stop_reason


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
    if state.debug and built.injected_blocks:
        _emit(f"(context: injected {', '.join(built.injected_blocks)})")

    if is_server_backend(state.backend_id):
        assert state.server_model is not None
        tts_on = state.backend_id == "lmstudio" and _tts_tools_enabled()
        sst_on = state.backend_id == "lmstudio" and _sst_tools_enabled()
        obs_on = state.backend_id == "lmstudio" and _obsidian_tools_wanted()
        shell_on = state.backend_id == "lmstudio" and _shell_tools_wanted()
        from cli.chat_tools import chat_tools_system_hint, default_chat_tools

        tools = default_chat_tools(
            tts_tool=tts_on,
            sst_tool=sst_on,
            obsidian_tool=obs_on,
            shell_tool=shell_on,
        )
        hint = chat_tools_system_hint(
            tts_tool=tts_on,
            sst_tool=sst_on,
            obsidian_tool=obs_on,
            shell_tool=shell_on,
        )
        if hint:
            has_system = any(m.get("role") == "system" for m in call_messages)
            if has_system:
                for m in call_messages:
                    if m.get("role") == "system":
                        prev = str(m.get("content") or "")
                        m["content"] = (prev + "\n\n" + hint).strip()
                        break
            else:
                call_messages = [{"role": "system", "content": hint}, *call_messages]
        try:
            prompt_tokens = 0
            completion_tokens = 0
            stop_reason = "stop"
            if tools:
                text, elapsed, spoke, prompt_tokens, completion_tokens, stop_reason = _run_server_tool_loop(
                    state,
                    call_messages,
                    tools=tools,
                )
            else:
                from backend.chat_resolve import server_chat_complete

                t0 = time.perf_counter()
                completion = server_chat_complete(
                    state.backend_id,
                    model=state.server_model,
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
        state.messages.append({"role": "assistant", "content": text})
        _persist_turn_after_success(state)
        _CURRENT_IO.on_assistant(text)
        if not spoke:
            _maybe_play_assistant_tts(state, text)
        state.stats.record_turn(
            input_tokens=prompt_tokens,
            new_tokens=completion_tokens,
            gen_time_s=elapsed,
            stop_reason=stop_reason,
        )
        if state.debug:
            _emit(state.stats.format_footer())
        return

    meta = state.meta
    assert meta is not None

    try:
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

    text = parsed_to_display_text(result.parsed)
    state.messages.append({"role": "assistant", "content": text})
    _persist_turn_after_success(state)
    _CURRENT_IO.on_assistant(text)

    _maybe_play_assistant_tts(state, text)

    state.stats.record_turn(
        input_tokens=result.input_tokens,
        new_tokens=result.new_tokens,
        gen_time_s=result.gen_time_s,
        stop_reason=result.stop_reason,
    )
    if state.debug:
        _emit(state.stats.format_footer())
    warn = state.stats.warn_if_context_high(0.9)
    if warn:
        _emit(warn)


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
    root = orodruin_project_root()
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

    retriever: RagRetriever | None = None
    try:
        candidate = load_rag_retriever(params.rag, native_index_path=params.rag_index)
        if candidate.backend_id() != "noop":
            retriever = candidate
    except Exception as exc:
        _emit(f"(retrieval init failed: {exc})")
        retriever = None

    memory_store = open_memory_store(params.memory_db)
    memory_scope: MemoryScope | None = None
    if memory_store is not None:
        session_id = params.memory_session or _dt.datetime.now().strftime("session_%Y%m%d_%H%M%S")
        memory_scope = MemoryScope(session_id=session_id, user_id=params.memory_user)

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
        retrieval_top_k=max(int(params.rag_top_k), 1),
        memory=memory_store,
        memory_scope=memory_scope,
        memory_recall_turns=max(int(params.memory_recall_turns), 0),
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
    )


def prepare_chat_session(
    params: ChatCliParams,
    *,
    on_load_progress: Callable[[int, int, str], None] | None = None,
) -> _SessionState:
    _suppress_noisy_warnings()

    enable_thinking = _thinking_default(params.thinking)
    root = orodruin_project_root()
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

    retriever: RagRetriever | None = None
    try:
        candidate = load_rag_retriever(params.rag, native_index_path=params.rag_index)
        if candidate.backend_id() != "noop":
            retriever = candidate
    except Exception as exc:
        _emit(f"(retrieval init failed: {exc})")
        retriever = None

    memory_store = open_memory_store(params.memory_db)
    memory_scope: MemoryScope | None = None
    if memory_store is not None:
        session_id = params.memory_session or _dt.datetime.now().strftime("session_%Y%m%d_%H%M%S")
        memory_scope = MemoryScope(session_id=session_id, user_id=params.memory_user)

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
        retrieval_top_k=max(int(params.rag_top_k), 1),
        memory=memory_store,
        memory_scope=memory_scope,
        memory_recall_turns=max(int(params.memory_recall_turns), 0),
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
    env = os.environ.get("ORODRUIN_CHAT_PRELOAD", "").strip().lower()
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
    root = orodruin_project_root()
    try:
        backend_id = resolve_chat_backend()
    except ValueError as exc:
        _emit(f"(backend resolve failed: {exc}; using hf)")
        backend_id = "hf"

    if is_server_backend(backend_id):
        server_model = _pick_default_server_model(backend_id)
        if server_model:
            _emit(f"(chat backend={backend_id}; model={server_model})")
        else:
            _emit(f"(chat backend={backend_id}; no models listed yet)")
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
    from cli.chat_display import chat_banner_lines

    return chat_banner_lines(state)


def close_chat_session(state: _SessionState) -> None:
    _unload_model_weights(state)
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
                continue
            handled, should_generate = dispatch_chat_line(state, raw_line)
            if state.exit_requested:
                break
            if handled and not should_generate:
                continue
            if not handled:
                state.messages.append({"role": "user", "content": raw_line.rstrip()})
            run_chat_generation(state)
    finally:
        if state is not None:
            close_chat_session(state)
        reset_chat_io()


def main() -> None:
    import sys as _sys

    from cli.terminal import chat_command as _chat

    _chat.main(args=_sys.argv[1:], prog_name="orodruin-chat-cli", standalone_mode=True)


if __name__ == "__main__":
    main()
