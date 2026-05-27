from __future__ import annotations

import os
import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

import click

try:
    from importlib.metadata import version as pkg_version_fn
except ImportError:
    pkg_version_fn = None

from backend.hf.paths import infer_default_quantization
from backend.hf.registry import preset_keys_sorted
from cli.chat import ChatCliParams, run_chat
from cli.inference import InferCliParams, run_infer
from cli.terminal_setup import TtsBackendChoice, finalize_tts
from processing.text.retrieval import rag_retriever_ids
from utils.device.env_bootstrap import load_mithril_dotenv


def _package_version_string() -> str:
    try:
        if pkg_version_fn is None:
            return "0"
        return str(pkg_version_fn("mithril"))
    except Exception:
        return "0"


def _validate_qbit(_ctx: click.Context, _param: click.Parameter, value: object) -> object:
    if value is None:
        return None
    if int(value) not in (0, 4, 8):
        raise click.BadParameter("expected one of {0,4,8} (GEMMA4_QBIT).")
    return int(value)


@click.group(invoke_without_command=False, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_package_version_string(), prog_name="mithril-cli")
def cli() -> None:
    load_mithril_dotenv()


@cli.command("infer", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("prompt", type=str)
@click.option(
    "--model",
    default=None,
    help=(
        "Local directory with config.json (defaults: GEMMA4_MODEL, GEMMA4_LOCAL_DIR built-in)."
    ),
)
@click.option(
    "--quantization",
    type=click.Choice(["none", "4bit", "8bit", "lightweight"]),
    default=infer_default_quantization(),
    help="Ignored when --qbit is set.",
)
@click.option(
    "--qbit",
    callback=_validate_qbit,
    default=None,
    type=int,
    help="0=none, 4=4bit, 8=8bit. Overrides --quantization. Env GEMMA4_QBIT.",
)
@click.option(
    "--system",
    default=os.environ.get("GEMMA4_SYSTEM_PROMPT"),
    help="Optional system message.",
)
@click.option("--max-new-tokens", default=512, type=int)
@click.option(
    "--thinking/--no-thinking",
    default=None,
    help="If omitted GEMMA4_THINKING controls the default.",
)
@click.option("--raw", is_flag=True, default=False, help="Do not strip special tokens from output.")
@click.option("--temperature", default=None, type=float)
@click.option("--top-p", default=None, type=float)
@click.option("--top-k", default=None, type=int)
@click.option("--repetition-penalty", default=None, type=float)
@click.option("--seed", default=None, type=int)
@click.option("--tts/--no-tts", default=False, help="Play assistant reply as speech after generation.")
@click.option("--tts-backend", default=None, type=TtsBackendChoice(), metavar="ID")
@click.option("--tts-model", default=None)
@click.option("--tts-ollama-model", default=None)
@click.option("--tts-speaker", default=None)
@click.option("--tts-language", default=None)
@click.option("--tts-instruct", default=None)
@click.option("--tts-max-chars", default=None, type=int)
@click.option("--tts-device", default=None)
@click.option("--tts-raw-output", is_flag=True, default=False)
def infer_command(
    prompt: str,
    model: str | None,
    quantization: str,
    qbit: int | None,
    system: str | None,
    max_new_tokens: int,
    thinking: bool | None,
    raw: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
    seed: int | None,
    tts: bool,
    tts_backend,
    tts_model: str | None,
    tts_ollama_model: str | None,
    tts_speaker: str | None,
    tts_language: str | None,
    tts_instruct: str | None,
    tts_max_chars: int | None,
    tts_device: str | None,
    tts_raw_output: bool,
) -> None:
    tts_opts = finalize_tts(
        tts_enabled=bool(tts),
        tts_backend=tts_backend,
        tts_model=tts_model,
        tts_ollama_model=tts_ollama_model,
        tts_speaker=tts_speaker,
        tts_language=tts_language,
        tts_instruct=tts_instruct,
        tts_max_chars=tts_max_chars,
        tts_device=tts_device,
        tts_raw_output=bool(tts_raw_output),
    )
    params = InferCliParams(
        prompt=prompt,
        model=model,
        quantization=quantization,
        qbit=qbit,
        system=system,
        max_new_tokens=max_new_tokens,
        thinking=thinking,
        raw=raw,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
        tts=tts_opts,
    )
    run_infer(params)


@cli.command("chat", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--model", default=None)
@click.option("--preset", type=click.Choice(tuple(preset_keys_sorted())), default=None)
@click.option(
    "--quantization",
    type=click.Choice(["none", "4bit", "8bit", "lightweight"]),
    default=infer_default_quantization(),
    help="Ignored when --qbit is set.",
)
@click.option(
    "--qbit",
    callback=_validate_qbit,
    default=None,
    type=int,
    help="0=none, 4=4bit, 8=8bit.",
)
@click.option(
    "--system",
    default=os.environ.get("GEMMA4_SYSTEM_PROMPT"),
)
@click.option("--max-new-tokens", default=512, type=int)
@click.option("--thinking/--no-thinking", default=None)
@click.option("--raw", is_flag=True, default=False)
@click.option("--debug/--no-debug", default=None, help="If omitted env MITHRIL_CHAT_DEBUG is used.")
@click.option("--temperature", default=None, type=float)
@click.option("--top-p", default=None, type=float)
@click.option("--top-k", default=None, type=int)
@click.option("--repetition-penalty", default=None, type=float)
@click.option("--seed", default=None, type=int)
@click.option(
    "--rag",
    type=click.Choice(tuple(sorted(list(rag_retriever_ids()), key=lambda s: str(s).lower()))),
    default=os.environ.get("MITHRIL_RAG", "noop"),
)
@click.option(
    "--rag-index",
    default=os.environ.get("MITHRIL_LEANN_INDEX") or None,
)
@click.option("--rag-top-k", default=int(os.environ.get("MITHRIL_RAG_TOP_K", "5")), type=int)
@click.option("--memory-db", default=os.environ.get("MITHRIL_MEMORY_DB") or None)
@click.option("--memory-session", default=os.environ.get("MITHRIL_MEMORY_SESSION") or None)
@click.option("--memory-user", default=os.environ.get("MITHRIL_MEMORY_USER") or None)
@click.option(
    "--memory-recall-turns",
    default=int(os.environ.get("MITHRIL_MEMORY_RECALL_TURNS", "6")),
    type=int,
)
@click.option("--tts/--no-tts", default=False)
@click.option("--tts-backend", default=None, type=TtsBackendChoice(), metavar="ID")
@click.option("--tts-model", default=None)
@click.option("--tts-ollama-model", default=None)
@click.option("--tts-speaker", default=None)
@click.option("--tts-language", default=None)
@click.option("--tts-instruct", default=None)
@click.option("--tts-max-chars", default=None, type=int)
@click.option("--tts-device", default=None)
@click.option("--tts-raw-output", is_flag=True, default=False)
@click.option(
    "--tui",
    is_flag=True,
    default=False,
    help="Run chat in a Textual TUI (requires pip install \"mithril[tui]\").",
)
@click.option(
    "--spawn-window/--no-spawn-window",
    default=True,
    show_default=True,
    help="With --tui, open a separate desktop terminal and mirror logs here.",
)
@click.option(
    "--interface",
    "chat_interface",
    type=click.Choice(["repl", "textual", "toad"], case_sensitive=False),
    default="textual",
    show_default=True,
    help="Chat UI host: textual (mithril TUI), repl (stdin), or toad (external Toad app).",
)
@click.option(
    "--chat-first",
    is_flag=True,
    default=False,
    help="With --tui, open chat mode first instead of the Textual home screen.",
)
def chat_command(
    model: str | None,
    preset: str | None,
    quantization: str,
    qbit: int | None,
    system: str | None,
    max_new_tokens: int,
    thinking: bool | None,
    raw: bool,
    debug: bool | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
    seed: int | None,
    rag: str,
    rag_index: str | None,
    rag_top_k: int,
    memory_db: str | None,
    memory_session: str | None,
    memory_user: str | None,
    memory_recall_turns: int,
    tts: bool,
    tts_backend,
    tts_model: str | None,
    tts_ollama_model: str | None,
    tts_speaker: str | None,
    tts_language: str | None,
    tts_instruct: str | None,
    tts_max_chars: int | None,
    tts_device: str | None,
    tts_raw_output: bool,
    tui: bool,
    spawn_window: bool,
    chat_interface: str,
    chat_first: bool,
) -> None:
    if (model is not None) and (preset is not None):
        raise click.UsageError("--model and --preset are mutually exclusive.")

    tts_opts = finalize_tts(
        tts_enabled=bool(tts),
        tts_backend=tts_backend,
        tts_model=tts_model,
        tts_ollama_model=tts_ollama_model,
        tts_speaker=tts_speaker,
        tts_language=tts_language,
        tts_instruct=tts_instruct,
        tts_max_chars=tts_max_chars,
        tts_device=tts_device,
        tts_raw_output=bool(tts_raw_output),
    )

    cfg = ChatCliParams(
        model=model,
        preset=preset,
        quantization=quantization,
        qbit=qbit,
        system=system,
        max_new_tokens=max_new_tokens,
        thinking=thinking,
        raw=raw,
        debug=debug,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
        rag=rag,
        rag_index=rag_index,
        rag_top_k=max(int(rag_top_k), 1),
        memory_db=memory_db,
        memory_session=memory_session,
        memory_user=memory_user,
        memory_recall_turns=max(int(memory_recall_turns), 0),
        tts=tts_opts,
        chat_first=bool(chat_first),
    )
    interface = str(chat_interface).lower()

    if interface == "toad":
        from cli.interface import launch_toad_session

        raise SystemExit(launch_toad_session(sys.argv[1:]))

    use_textual = bool(tui) or interface == "textual"
    if use_textual:
        from cli.interface import run_chat_interface

        run_chat_interface(
            interface="textual",
            params=cfg,
            user_argv=sys.argv[1:],
            spawn_window=spawn_window,
        )
        return

    run_chat(cfg)


def main() -> None:
    cli()


@cli.command("tui-profiles", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "action",
    type=click.Choice(["install", "status"], case_sensitive=False),
    default="install",
    required=False,
)
def tui_profiles_command(action: str) -> None:
    from cli.terminal_profiles import (
        install_terminal_profiles,
        profile_icon_path,
        profile_name,
        windows_profile_installed,
    )

    if action.lower() == "status":
        icon = profile_icon_path()
        print(f"profile name: {profile_name()}", flush=True)
        print(f"icon: {icon if icon else '(none)'}", flush=True)
        if sys.platform == "win32":
            print(f"windows profile installed: {windows_profile_installed()}", flush=True)
        return

    for line in install_terminal_profiles():
        print(line, flush=True)
