from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from backend.hf.paths import infer_default_quantization, resolve_cli_quantization
from eval.runner import RunConfig, TaskResult, run_task
from eval.task_spec import BenchmarkTask, find_task, list_task_ids, load_task
from utils.env_bootstrap import load_lmwrap_dotenv, lmwrap_project_root


def _default_data_dir() -> Path:
    env = os.environ.get("LMWRAP_BENCH_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (lmwrap_project_root() / "data" / "benchmarks").resolve()


def _default_out_dir() -> Path:
    env = os.environ.get("LMWRAP_BENCH_OUT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (lmwrap_project_root() / "evaluation_runs").resolve()


def _parse_task_ids(arg: str | None, data_dir: Path) -> list[str]:
    if not arg or arg.strip().lower() == "all":
        ids = list_task_ids(data_dir)
        if not ids:
            raise ValueError(f"no task manifests found under {data_dir}")
        return ids
    return [t.strip() for t in arg.split(",") if t.strip()]


def _load_tasks(task_ids: list[str], data_dir: Path) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    for tid in task_ids:
        spec = find_task(data_dir, tid)
        tasks.append(load_task(spec))
    return tasks


def _print_summary_table(results: list[TaskResult]) -> None:
    if not results:
        print("no tasks ran.")
        return
    width = max(len("task"), max(len(r.task_id) for r in results))
    print()
    print(f"{'task'.ljust(width)}  {'n':>5}  {'correct':>7}  {'acc':>6}  {'sec':>7}")
    print(f"{'-' * width}  {'-' * 5}  {'-' * 7}  {'-' * 6}  {'-' * 7}")
    for r in results:
        print(
            f"{r.task_id.ljust(width)}  {r.total:>5d}  {r.correct:>7d}  "
            f"{r.accuracy:>6.3f}  {r.elapsed_s:>7.1f}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmwrap-benchmark",
        description="Evaluate a local LLM on reasoning benchmarks defined in data/benchmarks.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task id, comma-separated list, or 'all'. Default: 'all' if --list is not used.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Folder with benchmark YAMLs (default: <repo>/data/benchmarks or LMWRAP_BENCH_DATA_DIR).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered task ids and exit.",
    )

    parser.add_argument(
        "--backend",
        choices=("hf", "ollama"),
        default=os.environ.get("LMWRAP_BENCH_BACKEND", "hf"),
        help="Inference backend (default: hf; override with LMWRAP_BENCH_BACKEND).",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="HF backend: local model directory (else GEMMA4_MODEL / GEMMA4_LOCAL_DIR).",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "4bit", "8bit", "lightweight"),
        default=infer_default_quantization(),
        help="HF backend: weights mode. Ignored when --qbit is set.",
    )
    parser.add_argument(
        "--qbit",
        type=int,
        choices=(0, 4, 8),
        default=None,
        help="HF backend short form: 0=none, 4=4bit, 8=8bit.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="HF backend: enable or disable built-in thinking mode (default: GEMMA4_THINKING env).",
    )

    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("LMWRAP_BENCH_OLLAMA_MODEL"),
        help="Ollama backend: model name visible to the local daemon.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Override Ollama base URL (else OLLAMA_HOST or http://127.0.0.1:11434).",
    )

    parser.add_argument("--limit", type=int, default=None, help="Cap examples per task.")
    parser.add_argument("--split", default=None, help="Source split (e.g. 'test' for MMLU).")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation budget.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override nucleus sampling top_p.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k sampling.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Override repetition penalty (>=1 discourages repetition).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Generation seed.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Run artifact root (default: <repo>/evaluation_runs or LMWRAP_BENCH_OUT_DIR).",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Subfolder name under <out-dir>/<task>/ (default: timestamp).",
    )
    return parser


def main() -> None:
    load_lmwrap_dotenv()
    args = _build_parser().parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else _default_data_dir()
    if args.list:
        ids = list_task_ids(data_dir)
        if not ids:
            print(f"(no task manifests under {data_dir})")
            return
        print(f"# task manifests under {data_dir}")
        for tid in ids:
            print(tid)
        return

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir()

    try:
        task_ids = _parse_task_ids(args.task, data_dir)
        tasks = _load_tasks(task_ids, data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"lmwrap-benchmark: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    enable_thinking = args.thinking
    if enable_thinking is None:
        enable_thinking = os.environ.get("GEMMA4_THINKING", "").lower() in ("1", "true", "yes")

    backend = args.backend
    if backend == "ollama" and not args.ollama_model:
        print(
            "lmwrap-benchmark: --backend ollama requires --ollama-model (or LMWRAP_BENCH_OLLAMA_MODEL).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    quantization = resolve_cli_quantization(qbit=args.qbit, quantization=args.quantization)

    results: list[TaskResult] = []
    for task in tasks:
        config = RunConfig(
            backend=backend,
            model_path=args.model,
            quantization=quantization,
            enable_thinking=bool(enable_thinking),
            ollama_model=args.ollama_model,
            ollama_base_url=args.ollama_host,
            max_new_tokens=(
                args.max_new_tokens
                if args.max_new_tokens is not None
                else task.defaults.max_new_tokens
            ),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            split=args.split,
            limit=args.limit,
            out_dir=out_dir,
            run_label=args.run_label,
        )
        try:
            results.append(run_task(task, config))
        except (FileNotFoundError, ValueError) as exc:
            print(f"lmwrap-benchmark: task {task.id} skipped: {exc}", file=sys.stderr)

    _print_summary_table(results)


if __name__ == "__main__":
    main()
