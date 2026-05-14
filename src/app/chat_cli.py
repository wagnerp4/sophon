from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from lmwrap.app.chat_stats import SessionStats, initial_debug_mode_from_env
from lmwrap.backend.gemma_backend import (
    GenerationResult,
    ModelMeta,
    generate_response,
    load_processor_and_model,
    parsed_to_display_text,
    read_model_meta,
)
from lmwrap.backend.gemma_paths import (
    infer_default_quantization,
    require_model_on_disk,
    resolve_cli_quantization,
    resolve_local_model_dir,
)
from lmwrap.utils.env_bootstrap import load_lmwrap_dotenv


@dataclass
class _GenParams:
    max_new_tokens: int = 512
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None


@dataclass
class _SessionState:
    processor: object
    model: object
    meta: ModelMeta
    messages: list[dict[str, object]] = field(default_factory=list)
    system_text: str | None = None
    enable_thinking: bool = False
    strip: bool = True
    debug: bool = False
    stats: SessionStats = field(default_factory=SessionStats)
    params: _GenParams = field(default_factory=_GenParams)
    exit_requested: bool = False


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
    print(msg, file=sys.stderr, flush=True)


def _format_help() -> str:
    rows: list[str] = ["commands:"]
    shortcut_for = {v: k for k, v in _SHORTCUTS.items()}
    for cmd in _COMMANDS.values():
        sc = shortcut_for.get(cmd.name)
        head = f"  /{cmd.name}" + (f" | {sc}" if sc else "")
        rows.append(f"{head:<22} {cmd.help_text}")
    rows.append('  """multi-line"""    Triple-quoted input is gathered until the closing """.')
    return "\n".join(rows)


@_register("help", "Show this help.", shortcut="?")
def _cmd_help(state: _SessionState, _arg: str) -> bool:
    _ = state
    _emit(_format_help())
    return False


@_register("quit", "Exit chat.", shortcut="q")
def _cmd_quit(state: _SessionState, _arg: str) -> bool:
    state.exit_requested = True
    return False


@_register("reset", "Clear conversation (keeps system prompt).")
def _cmd_reset(state: _SessionState, _arg: str) -> bool:
    state.messages = _baseline_messages(state.system_text)
    _emit("(conversation cleared)")
    return False


@_register("save", "Save transcript to <path> or chat_logs/<timestamp>.txt", shortcut="s")
def _cmd_save(state: _SessionState, arg: str) -> bool:
    target = arg.strip()
    if not target:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        target = str(Path("chat_logs") / f"{ts}.txt")
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
    _emit("(system updated, conversation reset)")
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


def _run_generation(state: _SessionState) -> None:
    cancelled = {"flag": False}
    prev_handler = signal.getsignal(signal.SIGINT)

    def _on_sigint(_signum: int, _frame: object) -> None:
        cancelled["flag"] = True
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        pass

    try:
        result: GenerationResult = generate_response(
            state.processor,
            state.model,
            state.messages,
            max_new_tokens=state.params.max_new_tokens,
            enable_thinking=state.enable_thinking,
            temperature=state.params.temperature,
            top_p=state.params.top_p,
            top_k=state.params.top_k,
            repetition_penalty=state.params.repetition_penalty,
            seed=state.params.seed,
            strip=state.strip,
            extra_specials=state.meta.special_tokens,
            eos_token_ids=state.meta.eos_token_ids or None,
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
    print(text)
    print(flush=True)

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat with local Hugging Face weights (Gemma / Llama / etc.).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local directory with config.json (default: GEMMA4_MODEL else GEMMA4_LOCAL_DIR or ./models/meta-llama-Llama-2-7b-chat-hf).",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit", "lightweight"],
        default=infer_default_quantization(),
        help="Weights: none=full; 4bit/8bit=CUDA quant; lightweight=same as --qbit 4. Ignored when --qbit is set.",
    )
    parser.add_argument(
        "--qbit",
        type=int,
        choices=[0, 4, 8],
        default=None,
        help="Short form: 0=none, 4=4bit, 8=8bit. Overrides --quantization when set. Env: GEMMA4_QBIT.",
    )
    parser.add_argument(
        "--system",
        default=os.environ.get("GEMMA4_SYSTEM_PROMPT"),
        help="Optional system message (omit if unset unless GEMMA4_SYSTEM_PROMPT is set).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Generation budget for new tokens per reply.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable built-in reasoning (thinking) mode. If omitted, GEMMA4_THINKING controls the default.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do not strip special tokens (e.g. </s>, channel markers) from displayed output.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print one-line stats footer after each reply. Env: LMWRAP_CHAT_DEBUG.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (>=1 discourages repetition).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Generation seed.")
    return parser


def main() -> None:
    load_lmwrap_dotenv()
    _suppress_noisy_warnings()

    parser = _build_parser()
    args = parser.parse_args()

    enable_thinking = _thinking_default(args.thinking)
    try:
        model_path = require_model_on_disk(resolve_local_model_dir(args.model))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    processor, model = load_processor_and_model(
        model_path,
        resolve_cli_quantization(qbit=args.qbit, quantization=args.quantization),
    )
    meta = read_model_meta(model_path, processor)

    debug = initial_debug_mode_from_env() if args.debug is None else bool(args.debug)
    params = _GenParams(
        max_new_tokens=int(args.max_new_tokens),
        temperature=args.temperature if args.temperature is not None else meta.default_temperature,
        top_p=args.top_p if args.top_p is not None else meta.default_top_p,
        top_k=args.top_k if args.top_k is not None else meta.default_top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    stats = SessionStats(max_position_embeddings=meta.max_position_embeddings)
    state = _SessionState(
        processor=processor,
        model=model,
        meta=meta,
        messages=_baseline_messages(args.system),
        system_text=args.system,
        enable_thinking=enable_thinking,
        strip=not bool(args.raw),
        debug=debug,
        stats=stats,
        params=params,
    )

    _emit(
        "lmwrap chat. Type /help or ? for commands. /quit or q to exit. EOF also leaves."
    )
    if meta.max_position_embeddings:
        _emit(f"(model context window: {meta.max_position_embeddings} tokens)")

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
        handled, should_generate = _dispatch_command(state, raw_line)
        if state.exit_requested:
            break
        if handled and not should_generate:
            continue
        if not handled:
            state.messages.append({"role": "user", "content": raw_line.rstrip()})
        _run_generation(state)


if __name__ == "__main__":
    main()
