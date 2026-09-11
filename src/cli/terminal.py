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
from cli.backends import dispatch, resolve_interface
from cli.chat import ChatCliParams
from cli.flags.tts import TtsBackendChoice, finalize_tts
from cli.flags.sst import SttBackendChoice, finalize_sst
from cli.inference import InferCliParams, run_infer
from processing.text.retrieval import rag_retriever_ids
from training.finetune.backends.factory import finetune_backend_ids
from training.finetune.datasets.registry import dataset_preset_keys_sorted
from utils.device.env_bootstrap import load_orodruin_dotenv


def _package_version_string() -> str:
    try:
        if pkg_version_fn is None:
            return "0"
        return str(pkg_version_fn("orodruin"))
    except Exception:
        return "0"


def build_chat_cli_params(
    *,
    model: str | None,
    preset: str | None,
    quantization: str | None,
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
    sst: bool,
    sst_backend,
    sst_model: str | None,
    sst_language: str | None,
    sst_device: str | None,
    sst_max_new_tokens: int | None,
    chat_first: bool,
    preload: bool,
) -> ChatCliParams:
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
    sst_opts = finalize_sst(
        sst_enabled=bool(sst),
        sst_backend=sst_backend,
        sst_model=sst_model,
        sst_language=sst_language,
        sst_device=sst_device,
        sst_max_new_tokens=sst_max_new_tokens,
    )

    return ChatCliParams(
        model=model,
        preset=preset,
        quantization=infer_default_quantization() if quantization is None else quantization,
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
        sst=sst_opts,
        chat_first=bool(chat_first),
        startup_preload=bool(preload),
    )


def parse_chat_argv(argv: list[str]) -> ChatCliParams:
    load_orodruin_dotenv()
    ctx = click.Context(chat_command, info_name="chat")
    with ctx.scope():
        chat_command.parse_args(ctx, list(argv))
        return build_chat_cli_params(
            model=ctx.params["model"],
            preset=ctx.params["preset"],
            quantization=ctx.params["quantization"],
            qbit=ctx.params["qbit"],
            system=ctx.params["system"],
            max_new_tokens=ctx.params["max_new_tokens"],
            thinking=ctx.params["thinking"],
            raw=ctx.params["raw"],
            debug=ctx.params["debug"],
            temperature=ctx.params["temperature"],
            top_p=ctx.params["top_p"],
            top_k=ctx.params["top_k"],
            repetition_penalty=ctx.params["repetition_penalty"],
            seed=ctx.params["seed"],
            rag=ctx.params["rag"],
            rag_index=ctx.params["rag_index"],
            rag_top_k=ctx.params["rag_top_k"],
            memory_db=ctx.params["memory_db"],
            memory_session=ctx.params["memory_session"],
            memory_user=ctx.params["memory_user"],
            memory_recall_turns=ctx.params["memory_recall_turns"],
            tts=ctx.params["tts"],
            tts_backend=ctx.params["tts_backend"],
            tts_model=ctx.params["tts_model"],
            tts_ollama_model=ctx.params["tts_ollama_model"],
            tts_speaker=ctx.params["tts_speaker"],
            tts_language=ctx.params["tts_language"],
            tts_instruct=ctx.params["tts_instruct"],
            tts_max_chars=ctx.params["tts_max_chars"],
            tts_device=ctx.params["tts_device"],
            tts_raw_output=ctx.params["tts_raw_output"],
            sst=ctx.params["sst"],
            sst_backend=ctx.params["sst_backend"],
            sst_model=ctx.params["sst_model"],
            sst_language=ctx.params["sst_language"],
            sst_device=ctx.params["sst_device"],
            sst_max_new_tokens=ctx.params["sst_max_new_tokens"],
            chat_first=ctx.params["chat_first"],
            preload=ctx.params["preload"],
        )


def _validate_qbit(_ctx: click.Context, _param: click.Parameter, value: object) -> object:
    if value is None:
        return None
    if int(value) not in (0, 4, 8):
        raise click.BadParameter("expected one of {0,4,8} (GEMMA4_QBIT).")
    return int(value)


@click.group(invoke_without_command=False, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_package_version_string(), prog_name="orodruin-cli")
def cli() -> None:
    load_orodruin_dotenv()


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
    default=None,
    help="Ignored when --qbit is set. Default: none, or GEMMA4_QBIT / GEMMA4_QUANTIZATION.",
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
    quantization: str | None,
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
        quantization=infer_default_quantization() if quantization is None else quantization,
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
    default=None,
    help="Ignored when --qbit is set. Default: none, or GEMMA4_QBIT / GEMMA4_QUANTIZATION.",
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
@click.option("--debug/--no-debug", default=None, help="If omitted env ORODRUIN_CHAT_DEBUG is used.")
@click.option("--temperature", default=None, type=float)
@click.option("--top-p", default=None, type=float)
@click.option("--top-k", default=None, type=int)
@click.option("--repetition-penalty", default=None, type=float)
@click.option("--seed", default=None, type=int)
@click.option(
    "--rag",
    type=click.Choice(tuple(sorted(list(rag_retriever_ids()), key=lambda s: str(s).lower()))),
    default=os.environ.get("ORODRUIN_RAG", "noop"),
)
@click.option(
    "--rag-index",
    default=os.environ.get("ORODRUIN_LEANN_INDEX") or None,
)
@click.option("--rag-top-k", default=int(os.environ.get("ORODRUIN_RAG_TOP_K", "5")), type=int)
@click.option("--memory-db", default=os.environ.get("ORODRUIN_MEMORY_DB") or None)
@click.option("--memory-session", default=os.environ.get("ORODRUIN_MEMORY_SESSION") or None)
@click.option("--memory-user", default=os.environ.get("ORODRUIN_MEMORY_USER") or None)
@click.option(
    "--memory-recall-turns",
    default=int(os.environ.get("ORODRUIN_MEMORY_RECALL_TURNS", "6")),
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
@click.option("--sst/--no-sst", default=False, help="Enable Qwen ASR for dropped audio files.")
@click.option("--sst-backend", default=None, type=SttBackendChoice(), metavar="ID")
@click.option("--sst-model", default=None)
@click.option("--sst-language", default=None)
@click.option("--sst-device", default=None)
@click.option("--sst-max-new-tokens", default=None, type=int)
@click.option(
    "--tui",
    is_flag=True,
    default=False,
    help="Alias for --interface textual (default interface).",
)
@click.option(
    "--spawn-window/--no-spawn-window",
    default=True,
    show_default=True,
    help="Open the Textual TUI in a separate desktop terminal and mirror logs here.",
)
@click.option(
    "--interface",
    "chat_interface",
    type=click.Choice(["repl", "textual", "toad"], case_sensitive=False),
    default="textual",
    show_default=True,
    help="Chat UI host: textual (orodruin TUI, default), repl (stdin), or toad (external Toad app).",
)
@click.option(
    "--chat-first",
    is_flag=True,
    default=False,
    help="Open chat mode first instead of the Textual home screen.",
)
@click.option(
    "--preload",
    is_flag=True,
    default=False,
    help="Load resolved local weights before the TUI opens (slow on WSL + /mnt/c).",
)
@click.option(
    "--windows/--linux",
    "use_windows_cli",
    default=None,
    help="WSL only: run via .venv/Scripts/orodruin-cli.exe (fast loads). Default: auto when .exe exists. Env ORODRUIN_WINDOWS_CLI.",
)
def chat_command(
    model: str | None,
    preset: str | None,
    quantization: str | None,
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
    sst: bool,
    sst_backend,
    sst_model: str | None,
    sst_language: str | None,
    sst_device: str | None,
    sst_max_new_tokens: int | None,
    tui: bool,
    spawn_window: bool,
    chat_interface: str,
    chat_first: bool,
    preload: bool,
    use_windows_cli: bool | None,
) -> None:
    from utils.device.platform import delegate_wsl_to_windows_cli, wsl_windows_cli_preference

    preference = wsl_windows_cli_preference(
        windows=use_windows_cli is True,
        linux=use_windows_cli is False,
    )
    delegate_wsl_to_windows_cli("orodruin-cli", sys.argv[1:], preference=preference)

    cfg = build_chat_cli_params(
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
        rag_top_k=rag_top_k,
        memory_db=memory_db,
        memory_session=memory_session,
        memory_user=memory_user,
        memory_recall_turns=memory_recall_turns,
        tts=tts,
        tts_backend=tts_backend,
        tts_model=tts_model,
        tts_ollama_model=tts_ollama_model,
        tts_speaker=tts_speaker,
        tts_language=tts_language,
        tts_instruct=tts_instruct,
        tts_max_chars=tts_max_chars,
        tts_device=tts_device,
        tts_raw_output=tts_raw_output,
        sst=sst,
        sst_backend=sst_backend,
        sst_model=sst_model,
        sst_language=sst_language,
        sst_device=sst_device,
        sst_max_new_tokens=sst_max_new_tokens,
        chat_first=chat_first,
        preload=preload,
    )
    interface = resolve_interface(tui=tui, chat_interface=chat_interface)
    dispatch(
        interface,
        cfg,
        spawn_window=spawn_window,
        user_argv=sys.argv[1:],
    )


@cli.command("finetune", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--preset",
    required=True,
    type=click.Choice(tuple(preset_keys_sorted())),
    help="Base model preset key.",
)
@click.option(
    "--dataset",
    default="gsm8k_instructions",
    type=click.Choice(tuple(dataset_preset_keys_sorted())),
    show_default=True,
    help="Finetune dataset preset.",
)
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(tuple(finetune_backend_ids())),
    show_default=True,
    help="Training backend.",
)
@click.argument("overrides", nargs=-1)
def finetune_command(preset: str, dataset: str, backend: str, overrides: tuple[str, ...]) -> None:
    from training.finetune.job import run_finetune_job
    from utils.device.env_bootstrap import orodruin_project_root

    def on_log(msg: str) -> None:
        click.echo(f"[orodruin] {msg}")

    def on_progress(step: int, total: int, label: str) -> None:
        if total > 0:
            click.echo(f"[orodruin] {label} ({step}/{total})")
        else:
            click.echo(f"[orodruin] {label}")

    try:
        result = run_finetune_job(
            preset,
            dataset,
            project_root=orodruin_project_root(),
            backend_id=backend,
            recipe_override_tokens=list(overrides),
            on_progress=on_progress,
            on_log=on_log,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"adapter saved to {result.adapter_dir}")


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
    from cli.host.profiles import (
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
