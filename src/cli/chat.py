from __future__ import annotations

import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

# Todo: remove this path bootstrap after the package uses consistent mithril.* imports end-to-end.

import datetime as _dt
import logging
import os
import signal
import time
import warnings
from dataclasses import dataclass, field
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

from backend.hf.paths import (
    infer_default_quantization,
    require_model_on_disk,
    resolve_cli_quantization,
    resolve_local_model_dir,
)
from processing.text.context import ContextBuildResult, build_messages_for_model
from processing.text.memory import MemoryScope, MemoryStore, open_memory_store
from processing.text.retrieval import RagRetriever, RetrievalQuery, RetrievalResult, load_rag_retriever, rag_retriever_ids
from processing.audio.speech.cli import TtsCliOptions
from processing.audio.speech.factory import create_speech_tts_engine
from processing.audio.speech.protocols import SpeechTtsEngine
from processing.audio.speech.speak import speak_text_blocking
from processing.audio.speech.types import TtsBackendId, normalize_tts_backend_token, tts_backend_help_tokens
from utils.device.env_bootstrap import mithril_chat_logs_dir, mithril_project_root
from backend.hf.registry import (
    HF_MODEL_PRESETS,
    local_only_model_dirs,
    match_preset_key,
    preset_has_weights,
    preset_keys_sorted,
    resolve_preset_dir,
)
from utils.download.hf import download_preset_snapshot


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
    chat_first: bool = False


@dataclass(frozen=True)
class ChatIo:
    emit: Callable[[str], None]
    on_assistant: Callable[[str], None]


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
    tts_speaker: str = "Ryan"
    tts_language: str = "English"
    tts_instruct: str | None = None
    tts_max_chars: int = 8000
    tts_device: str | None = None
    tts_backend_id: TtsBackendId = "hf_qwen_custom_voice"
    tts_ollama_model: str | None = None
    tts_engine: SpeechTtsEngine | None = None
    model_path: str = ""
    preset_key: str | None = None
    quantization: str = "none"
    project_root: Path = field(default_factory=Path.cwd)
    model_switching: bool = False


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


def _register(name: str, help_text: str, shortcut: str | None = None):
    def deco(fn: Callable[["_SessionState", str], bool]) -> Callable[["_SessionState", str], bool]:
        _COMMANDS[name] = _Command(name=name, help_text=help_text, handler=fn)
        if shortcut:
            _SHORTCUTS[shortcut] = name
        return fn

    return deco


def _emit(msg: str) -> None:
    _CURRENT_IO.emit(msg)


def _format_help() -> str:
    rows: list[str] = ["commands:"]
    shortcut_for = {v: k for k, v in _SHORTCUTS.items()}
    for cmd in _COMMANDS.values():
        sc = shortcut_for.get(cmd.name)
        head = f"  /{cmd.name}" + (f" | {sc}" if sc else "")
        rows.append(f"{head:<22} {cmd.help_text}")
    rows.append("  \"\"\"multi-line\"\"\"    Triple-quoted input is gathered until the closing \"\"\".")
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


def build_model_picker_entries(state: _SessionState) -> list[ModelPickerEntry]:
    root = state.project_root
    current_preset = state.preset_key
    current_path = Path(state.model_path).resolve() if state.model_path.strip() else None

    downloaded = [key for key in preset_keys_sorted() if preset_has_weights(key, root)]
    missing = [key for key in preset_keys_sorted() if not preset_has_weights(key, root)]
    extra_dirs = local_only_model_dirs(root)

    entries: list[ModelPickerEntry] = []
    if downloaded or extra_dirs:
        entries.append(ModelPickerEntry("", "local", "", False, False, "header"))
        for key in downloaded:
            preset = HF_MODEL_PRESETS[key]
            is_current = key == current_preset or (
                current_path is not None and resolve_preset_dir(key, root).resolve() == current_path
            )
            entries.append(
                ModelPickerEntry(
                    target=key,
                    title=key,
                    detail=preset.repo_id,
                    local=True,
                    current=is_current,
                    kind="preset",
                )
            )
        for path in extra_dirs:
            is_current = current_path is not None and path.resolve() == current_path
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
        entries.append(ModelPickerEntry("", "not downloaded", "", False, False, "header"))
        for key in missing:
            preset = HF_MODEL_PRESETS[key]
            entries.append(
                ModelPickerEntry(
                    target=key,
                    title=key,
                    detail=preset.repo_id,
                    local=False,
                    current=False,
                    kind="preset",
                )
            )
    return entries


def _resolve_model_target(token: str, root: Path) -> tuple[str | None, Path]:
    raw = token.strip()
    if not raw:
        raise ValueError("empty model target")
    try:
        preset_key = match_preset_key(raw)
    except ValueError:
        raise
    if preset_key is not None:
        return preset_key, resolve_preset_dir(preset_key, root)
    path = Path(raw).expanduser()
    if path.exists() or any(sep in raw for sep in ("/", "\\", ":")):
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

    root = state.project_root
    current_preset = state.preset_key
    current_path = Path(state.model_path).resolve() if state.model_path.strip() else None

    downloaded: list[str] = []
    missing: list[str] = []
    for key in preset_keys_sorted():
        if preset_has_weights(key, root):
            downloaded.append(key)
        else:
            missing.append(key)

    lines: list[str] = []
    if not chat_session_has_weights(state):
        lines.append("status: no model weights loaded (chat generation disabled until /model or panel load)")
    if current_preset:
        lines.append(f"current preset: {current_preset}")
        lines.append(f"current path:   {current_path if current_path is not None else '(none)'}")
    elif current_path is not None:
        lines.append(f"current path:   {current_path}")
    else:
        lines.append("current path:   (none)")
    lines.append(
        f"presets: {len(downloaded)} local, {len(missing)} not downloaded "
        f"({len(HF_MODEL_PRESETS)} registered)"
    )

    def append_preset_block(title: str, keys: list[str]) -> None:
        if not keys:
            return
        lines.append("")
        lines.append(title)
        for key in keys:
            preset = HF_MODEL_PRESETS[key]
            is_current = key == current_preset or (
                current_path is not None and resolve_preset_dir(key, root).resolve() == current_path
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
        append_preset_block("local presets:", downloaded)
    if view_norm in ("all", "missing", "remote"):
        append_preset_block("not downloaded:", missing)

    extra_dirs = local_only_model_dirs(root)
    if extra_dirs and view_norm in ("all", "local", "downloaded"):
        lines.append("")
        lines.append("local paths (no preset key):")
        for path in extra_dirs:
            marker = "*" if current_path is not None and path.resolve() == current_path else " "
            lines.append(_format_model_line(marker=marker, label=str(path.name), repo_or_path=str(path)))

    return "\n".join(lines)


def _unload_model_weights(state: _SessionState) -> None:
    state.processor = None
    state.model = None
    state.tts_engine = None
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
    preset_key, model_dir = _resolve_model_target(target, state.project_root)
    if not (model_dir / "config.json").is_file():
        if preset_key is not None:
            _emit(
                f"preset {preset_key!r} is not on disk at {model_dir}. "
                f"Run /model-download {preset_key}"
            )
        else:
            _emit(f"model directory missing config.json: {model_dir}")
        return

    model_path = str(model_dir.resolve())
    if model_path == state.model_path and preset_key == state.preset_key:
        _emit(f"(already using {preset_key or model_path})")
        return

    state.model_switching = True
    try:
        _emit(f"loading model from {model_path} ...")
        processor, model = load_processor_and_model(
            model_path,
            state.quantization,
            on_load_progress=on_load_progress,
        )
        meta = read_model_meta(model_path, processor)
    except Exception as exc:
        _emit(f"(model load failed: {exc})")
        return
    finally:
        state.model_switching = False

    old_processor = state.processor
    old_model = state.model
    state.processor = processor
    state.model = model
    del old_processor
    del old_model
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

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
    _emit(f"(switched to {label}{ctx_note}; conversation cleared)")


def _current_model_label(state: _SessionState) -> str:
    if state.preset_key:
        preset = HF_MODEL_PRESETS[state.preset_key]
        return f"preset={state.preset_key} repo={preset.repo_id} path={state.model_path}"
    if state.model_path.strip():
        return f"path={state.model_path}"
    return "(no model directory; use /models or /model PATH)"


@_register("help", "Show this help.", shortcut="?")
def _cmd_help(state: _SessionState, _arg: str) -> bool:
    _ = state
    _emit(_format_help())
    return False


@_register("models", "List presets: /models [all | local | missing].")
def _cmd_models(state: _SessionState, arg: str) -> bool:
    _emit(format_model_catalog(state, view=arg.strip() or "all"))
    return False


@_register("model", "Show or switch model: /model [PRESET | PATH].")
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
    if preset_has_weights(preset_key, state.project_root):
        _emit(f"(preset {preset_key!r} already on disk)")
        return False
    preset = HF_MODEL_PRESETS[preset_key]
    _emit(f"downloading {preset_key} ({preset.repo_id}) ...")
    state.model_switching = True
    try:
        path = download_preset_snapshot(preset_key, verbose=True)
    except Exception as exc:
        _emit(f"(download failed: {exc})")
        return False
    finally:
        state.model_switching = False
    _emit(f"(downloaded to {path}; use /model {preset_key} to load)")
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
        target = str(mithril_chat_logs_dir() / f"{ts}.txt")
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


@_register("rag-status", "Show retrieval backend status.")
def _cmd_rag_status(state: _SessionState, _arg: str) -> bool:
    if state.retriever is None:
        _emit("retrieval: disabled")
    else:
        _emit(f"retrieval: backend={state.retriever.backend_id()} top_k={state.retrieval_top_k}")
    return False


@_register("memory-status", "Show memory store and session info.")
def _cmd_memory_status(state: _SessionState, _arg: str) -> bool:
    if state.memory is None or state.memory_scope is None:
        _emit("memory: disabled (pass --memory-db or set MITHRIL_MEMORY_DB)")
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
            f"plain={state.tts_plain_text} hf_model={state.tts_model_id!r} "
            f"ollama_model={om} instruct={ins}"
        )
    else:
        _emit("usage: /tts on | off | show")
    return False


@_register("tts-speaker", "Set CustomVoice speaker name (e.g. Ryan, Aiden, Vivian).")
def _cmd_tts_speaker(state: _SessionState, arg: str) -> bool:
    if not arg.strip():
        _emit(f"tts speaker = {state.tts_speaker!r}")
        return False
    state.tts_speaker = arg.strip()
    _emit(f"(tts speaker = {state.tts_speaker!r})")
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


def _run_retrieval(state: _SessionState) -> RetrievalResult | None:
    if state.retriever is None:
        return None
    query_text = _latest_user_text(state)
    if not query_text:
        return None
    try:
        return state.retriever.retrieve(
            RetrievalQuery(text=query_text, params={"top_k": state.retrieval_top_k}),
        )
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


def _maybe_play_assistant_tts(state: _SessionState, text: str) -> None:
    if not state.tts_enabled:
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
    )


def _run_generation(state: _SessionState) -> None:
    if not chat_session_has_weights(state):
        _emit(
            "No model weights are loaded. Use /models to list presets, /model-download PRESET to fetch one, "
            "or /model PATH for a local directory with config.json."
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
) -> _SessionState:
    _suppress_noisy_warnings()

    enable_thinking = _thinking_default(params.thinking)
    root = mithril_project_root()
    quantization = resolve_cli_quantization(qbit=params.qbit, quantization=params.quantization)
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
        model_path=model_path_str,
        preset_key=preset_key,
        quantization=quantization,
        project_root=root,
    )


def prepare_chat_session(
    params: ChatCliParams,
    *,
    on_load_progress: Callable[[int, int, str], None] | None = None,
) -> _SessionState:
    _suppress_noisy_warnings()

    enable_thinking = _thinking_default(params.thinking)
    root = mithril_project_root()
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
        model_path=model_path,
        preset_key=preset_key,
        quantization=quantization,
        project_root=root,
    )


def prepare_chat_session_or_shell(
    params: ChatCliParams,
    *,
    on_load_progress: Callable[[int, int, str], None] | None = None,
) -> _SessionState:
    root = mithril_project_root()
    preset_key = params.preset
    if params.preset is not None:
        resolved_model_dir = resolve_preset_dir(params.preset, root)
    else:
        resolved_model_dir = resolve_local_model_dir(params.model)
    if (resolved_model_dir / "config.json").is_file():
        return prepare_chat_session(params, on_load_progress=on_load_progress)
    return prepare_shell_chat_session(params, resolved_model_dir, preset_key)


def chat_startup_lines(state: _SessionState) -> list[str]:
    lines = ["mithril chat. Type /help or ? for commands. /quit or q to exit."]
    if not chat_session_has_weights(state):
        lines.append(
            "(warning: no weights loaded. Chat generation is disabled until /model-download PRESET, "
            "/model PATH, or picking a model in the panel.)"
        )
    if state.preset_key:
        preset = HF_MODEL_PRESETS[state.preset_key]
        lines.append(f"(model: {state.preset_key} — {preset.repo_id})")
    elif state.model_path.strip():
        lines.append(f"(model path: {state.model_path})")
    meta = state.meta
    if meta is not None and meta.max_position_embeddings:
        lines.append(f"(model context window: {meta.max_position_embeddings} tokens)")
    if state.retriever is not None:
        lines.append(f"(retrieval: {state.retriever.backend_id()} top_k={state.retrieval_top_k})")
    if state.memory is not None and state.memory_scope is not None:
        lines.append(
            f"(memory: session={state.memory_scope.session_id} "
            f"recall_turns={state.memory_recall_turns})"
        )
    if state.tts_enabled:
        if state.tts_backend_id == "hf_qwen_custom_voice":
            lines.append(
                "(tts: HF Qwen3-TTS CustomVoice; install extras with pip install \"mithril[tts]\")"
            )
        else:
            lines.append("(tts: Ollama backend selected; synthesis wiring still pending)")
    return lines


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

    _chat.main(args=_sys.argv[1:], prog_name="mithril-chat-cli", standalone_mode=True)


if __name__ == "__main__":
    main()
