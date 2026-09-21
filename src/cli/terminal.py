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
from utils.device.env_bootstrap import load_sophon_dotenv


def _resolve_rag_cli_defaults(rag: str | None, rag_index: str | None) -> tuple[str, str | None]:
    from pathlib import Path

    from processing.text.retrieval.corpus import default_index_exists, default_leann_index_path

    rag_env = os.environ.get("SOPHON_RAG", "").strip()
    index_env = os.environ.get("SOPHON_LEANN_INDEX", "").strip()
    index = rag_index if rag_index else (index_env or None)
    default_path = default_leann_index_path()
    if not index and default_index_exists(default_path):
        index = str(default_path)
    if rag is not None and str(rag).strip():
        backend = str(rag).strip()
    elif rag_env:
        backend = rag_env
    elif index and default_index_exists(Path(index)):
        backend = "leann"
    else:
        backend = "noop"
    return backend, index


def _package_version_string() -> str:
    try:
        if pkg_version_fn is None:
            return "0"
        return str(pkg_version_fn("sophon"))
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
    rag: str | None,
    rag_index: str | None,
    rag_top_k: int,
    rag_adaptive: bool,
    rag_structure: str,
    rag_structure_dir: str | None,
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
    resolved_rag, resolved_index = _resolve_rag_cli_defaults(rag, rag_index)
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
        rag=resolved_rag,
        rag_index=resolved_index,
        rag_top_k=max(int(rag_top_k), 1),
        rag_adaptive=bool(rag_adaptive),
        rag_structure=str(rag_structure or "none"),
        rag_structure_dir=rag_structure_dir,
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
    load_sophon_dotenv()
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
            rag_adaptive=ctx.params["rag_adaptive"],
            rag_structure=ctx.params["rag_structure"],
            rag_structure_dir=ctx.params["rag_structure_dir"],
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
@click.version_option(_package_version_string(), prog_name="sophon-cli")
def cli() -> None:
    load_sophon_dotenv()


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
@click.option("--max-new-tokens", default=2048, type=int)
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
@click.option("--max-new-tokens", default=2048, type=int)
@click.option("--thinking/--no-thinking", default=None)
@click.option("--raw", is_flag=True, default=False)
@click.option("--debug/--no-debug", default=None, help="If omitted env SOPHON_CHAT_DEBUG is used.")
@click.option("--temperature", default=None, type=float)
@click.option("--top-p", default=None, type=float)
@click.option("--top-k", default=None, type=int)
@click.option("--repetition-penalty", default=None, type=float)
@click.option("--seed", default=None, type=int)
@click.option(
    "--rag",
    type=click.Choice(tuple(sorted(list(rag_retriever_ids()), key=lambda s: str(s).lower()))),
    default=None,
    help="Retrieval backend. Default: leann when data/rag/indexes/default exists, else noop. Env: SOPHON_RAG.",
)
@click.option(
    "--rag-index",
    default=None,
    help="LEANN index path. Default: data/rag/indexes/default when present. Env: SOPHON_LEANN_INDEX.",
)
@click.option("--rag-top-k", default=int(os.environ.get("SOPHON_RAG_TOP_K", "5")), type=int)
@click.option(
    "--rag-adaptive/--no-rag-adaptive",
    default=os.environ.get("SOPHON_RAG_ADAPTIVE", "1").strip().lower() not in ("0", "false", "no", "off"),
    help="Adaptive-RAG gate: skip / single-hop / multi-hop before retrieve.",
)
@click.option(
    "--rag-structure",
    type=click.Choice(["none", "lightrag"]),
    default=os.environ.get("SOPHON_RAG_STRUCTURE", "none"),
    help="Graph/structure retriever for multi-hop queries (LightRAG).",
)
@click.option(
    "--rag-structure-dir",
    default=os.environ.get("SOPHON_LIGHTRAG_DIR") or None,
    help="Working directory for LightRAG storage. Env: SOPHON_LIGHTRAG_DIR.",
)
@click.option(
    "--memory-db",
    default=os.environ.get("SOPHON_MEMORY_DB") or None,
    help="SQLite path. Default: data/memory/memory.db. Env: SOPHON_MEMORY_DB. Set 0 to disable.",
)
@click.option("--memory-session", default=os.environ.get("SOPHON_MEMORY_SESSION") or None)
@click.option("--memory-user", default=os.environ.get("SOPHON_MEMORY_USER") or None)
@click.option(
    "--memory-recall-turns",
    default=int(os.environ.get("SOPHON_MEMORY_RECALL_TURNS", "6")),
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
    help="Chat UI host: textual (sophon TUI, default), repl (stdin), or toad (external Toad app).",
)
@click.option(
    "--chat-first/--dashboard-first",
    default=True,
    show_default=True,
    help="Open Nexus (chat) first. Use --dashboard-first for the dashboard home screen.",
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
    help="WSL only: run via .venv/Scripts/sophon-cli.exe (fast loads). Default: auto when .exe exists. Env SOPHON_WINDOWS_CLI.",
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
    rag: str | None,
    rag_index: str | None,
    rag_top_k: int,
    rag_adaptive: bool,
    rag_structure: str,
    rag_structure_dir: str | None,
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
    delegate_wsl_to_windows_cli("sophon-cli", sys.argv[1:], preference=preference)

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
        rag_adaptive=rag_adaptive,
        rag_structure=rag_structure,
        rag_structure_dir=rag_structure_dir,
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
    default=None,
    type=click.Choice(tuple(preset_keys_sorted())),
    help="Base model preset key. Required unless --recipe is set.",
)
@click.option(
    "--dataset",
    default=None,
    type=click.Choice(tuple(dataset_preset_keys_sorted())),
    help="Finetune dataset preset. Default gsm8k_instructions when --recipe is omitted.",
)
@click.option(
    "--recipe",
    "recipe_path",
    default=None,
    type=click.Path(),
    help="YAML recipe path (config/finetune/....yaml).",
)
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(tuple(finetune_backend_ids())),
    show_default=True,
    help="Training backend.",
)
@click.argument("overrides", nargs=-1)
def finetune_command(
    preset: str | None,
    dataset: str | None,
    recipe_path: str | None,
    backend: str,
    overrides: tuple[str, ...],
) -> None:
    from training.finetune.job import run_finetune_job
    from utils.device.env_bootstrap import sophon_project_root

    if recipe_path is None and preset is None:
        raise click.UsageError("--preset is required unless --recipe is set")
    dataset_id = dataset
    if recipe_path is None and dataset_id is None:
        dataset_id = "gsm8k_instructions"

    def on_log(msg: str) -> None:
        click.echo(f"[sophon] {msg}")

    def on_progress(step: int, total: int, label: str) -> None:
        if total > 0:
            click.echo(f"[sophon] {label} ({step}/{total})")
        else:
            click.echo(f"[sophon] {label}")

    try:
        result = run_finetune_job(
            preset,
            dataset_id,
            project_root=sophon_project_root(),
            backend_id=backend,
            recipe_path=recipe_path,
            recipe_override_tokens=list(overrides),
            on_progress=on_progress,
            on_log=on_log,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"adapter saved to {result.adapter_dir}")
    if result.adapter_name:
        click.echo(f"/adapter load {result.adapter_name}")


def main() -> None:
    cli()


@cli.command("rag-index", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--vault",
    "vault_root",
    default=None,
    help="Obsidian vault filesystem root. Env: SOPHON_VAULT_PATH.",
)
@click.option(
    "--project/--no-project",
    "include_project",
    default=True,
    show_default=True,
    help="Include the Sophon project tree in the corpus.",
)
@click.option(
    "--project-root",
    default=None,
    help="Override project root (default: package project root).",
)
@click.option(
    "--index",
    "index_path",
    default=None,
    help="LEANN index path (default: data/rag/indexes/default or SOPHON_LEANN_INDEX).",
)
@click.option(
    "--structure-dir",
    default=None,
    help="Optional LightRAG working directory (env: SOPHON_LIGHTRAG_DIR).",
)
@click.option("--rebuild", is_flag=True, default=False, help="Wipe existing LEANN index files first.")
@click.option("--chunk-chars", default=1200, type=int, show_default=True)
@click.option("--overlap", default=150, type=int, show_default=True)
@click.option("--embedding-model", default=None, help="Optional LEANN embedding model override.")
def rag_index_cli(
    vault_root: str | None,
    include_project: bool,
    project_root: str | None,
    index_path: str | None,
    structure_dir: str | None,
    rebuild: bool,
    chunk_chars: int,
    overlap: int,
    embedding_model: str | None,
) -> None:
    from cli.rag_index import rag_index_command

    ctx = click.get_current_context()
    ctx.invoke(
        rag_index_command,
        vault_root=vault_root,
        include_project=include_project,
        project_root=project_root,
        index_path=index_path,
        structure_dir=structure_dir,
        rebuild=rebuild,
        chunk_chars=chunk_chars,
        overlap=overlap,
        embedding_model=embedding_model,
    )


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
    from utils.device.platform import is_wsl

    if action.lower() == "status":
        icon = profile_icon_path()
        print(f"profile name: {profile_name()}", flush=True)
        print(f"icon: {icon if icon else '(none)'}", flush=True)
        if sys.platform == "win32" or is_wsl():
            print(f"windows profile installed: {windows_profile_installed()}", flush=True)
        return

    for line in install_terminal_profiles():
        print(line, flush=True)


@cli.command("deploy-windows", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--sync-venv/--skip-venv",
    default=None,
    help="Run uv sync on the Windows deploy tree via Store pwsh. Default: only if sophon-chat-tui.exe is missing.",
)
@click.option(
    "--profile-only",
    is_flag=True,
    default=False,
    help="Rewrite the Windows Terminal sophon profile without rsync or uv sync.",
)
def deploy_windows_command(sync_venv: bool | None, profile_only: bool) -> None:
    from cli.host.windows_deploy import deploy_windows
    from utils.device.env_bootstrap import load_sophon_dotenv

    load_sophon_dotenv()
    try:
        lines = deploy_windows(
            sync_files=not profile_only,
            sync_venv=False if profile_only else sync_venv,
            install_profile=True,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    for line in lines:
        click.echo(line)
