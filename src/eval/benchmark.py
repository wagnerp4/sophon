from __future__ import annotations

import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

# Todo: remove this path bootstrap after the package uses consistent orodruin.* imports end-to-end.

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

from backend.hf.paths import (
    infer_default_quantization,
    resolve_cli_quantization,
    resolve_local_model_dir,
)
from eval.experiment_id import (
    BenchFingerprintParts,
    default_run_id,
    fingerprint_hash,
    fingerprint_payload,
    sanitize_run_id_component,
    slugify_segment,
)
from eval.runner import RunConfig, TaskResult, run_task
from eval.task_spec import BenchmarkTask, find_task, list_task_ids, load_task
from utils.device.env_bootstrap import load_orodruin_dotenv, orodruin_project_root
from backend.hf.registry import preset_keys_sorted, resolve_preset_dir


def _default_data_dir() -> Path:
    env = os.environ.get("ORODRUIN_BENCH_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (orodruin_project_root() / "data" / "benchmarks").resolve()


def _default_out_dir() -> Path:
    env = os.environ.get("ORODRUIN_BENCH_OUT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (orodruin_project_root() / "data" / "exps").resolve()


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
        prog="orodruin-benchmark",
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
        help="Folder with benchmark YAMLs (default: <repo>/data/benchmarks or ORODRUIN_BENCH_DATA_DIR).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered task ids and exit.",
    )

    parser.add_argument(
        "--backend",
        choices=("hf", "ollama"),
        default=os.environ.get("ORODRUIN_BENCH_BACKEND", "hf"),
        help="Inference backend (default: hf; override with ORODRUIN_BENCH_BACKEND).",
    )

    preset_choices = preset_keys_sorted()
    mx_model = parser.add_mutually_exclusive_group()
    mx_model.add_argument(
        "--model",
        default=None,
        help="HF backend: local model directory. Mutually exclusive with --preset. Else GEMMA4_MODEL / GEMMA4_LOCAL_DIR.",
    )
    mx_model.add_argument(
        "--preset",
        choices=preset_choices,
        default=None,
        help="HF backend: registry key; loads <repo>/models/<Hub-dash-id> (like orodruin-hf-download). Mutually exclusive with --model.",
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
        default=os.environ.get("ORODRUIN_BENCH_OLLAMA_MODEL"),
        help="Ollama backend: model name visible to the local daemon.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Override Ollama base URL (else OLLAMA_HOST or http://127.0.0.1:11434).",
    )

    parser.add_argument("--limit", type=int, default=None, help="Cap examples per task.")
    parser.add_argument(
        "--split",
        default=None,
        help="Source split where supported (MMLU dev/test; Hellaswag val/train).",
    )
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
        help="Experiment root folder (default: <repo>/data/exps or ORODRUIN_BENCH_OUT_DIR).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Experiment folder name under --out-dir (default: timestamp + backend + model + quant + thinking + config digest).",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional suffix appended to the run id as __<label> for notes.",
    )
    return parser


def main() -> None:
    load_orodruin_dotenv()
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
        print(f"orodruin-benchmark: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    enable_thinking = args.thinking
    if enable_thinking is None:
        enable_thinking = os.environ.get("GEMMA4_THINKING", "").lower() in ("1", "true", "yes")

    backend = args.backend
    if args.preset is not None and backend != "hf":
        print(
            "orodruin-benchmark: --preset applies only with --backend hf.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    root = orodruin_project_root()
    effective_hf_model = args.model
    if args.preset is not None:
        effective_hf_model = str(resolve_preset_dir(args.preset, root))

    if backend == "ollama" and not args.ollama_model:
        print(
            "orodruin-benchmark: --backend ollama requires --ollama-model (or ORODRUIN_BENCH_OLLAMA_MODEL).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    quantization = resolve_cli_quantization(qbit=args.qbit, quantization=args.quantization)

    model_slug = slugify_segment(args.ollama_model or "unknown")
    resolved_model_display = None
    if backend == "hf":
        resolved = resolve_local_model_dir(effective_hf_model)
        resolved_model_display = str(resolved.resolve())
        digest = hashlib.sha256(resolved_model_display.encode("utf-8")).hexdigest()[:8]
        model_slug = f"{slugify_segment(resolved.name)}_{digest}"

    parts = BenchFingerprintParts(
        backend=backend,
        model_slug=model_slug,
        quantization=quantization,
        enable_thinking=bool(enable_thinking),
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_host,
        max_new_tokens_override=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        split=args.split,
        limit=args.limit,
    )
    explicit_rid = (args.run_id or "").strip()
    if explicit_rid:
        run_id = sanitize_run_id_component(explicit_rid)
    else:
        run_id = default_run_id(parts)
    if args.run_label:
        run_id = sanitize_run_id_component(f"{run_id}__{slugify_segment(args.run_label, 48)}")

    exp_root = (out_dir / run_id).resolve()
    exp_root.mkdir(parents=True, exist_ok=True)
    fp_payload = fingerprint_payload(parts)
    print(f"orodruin-benchmark: run_id={run_id}", file=sys.stderr)
    print(f"orodruin-benchmark: experiment_dir={exp_root}", file=sys.stderr)
    if resolved_model_display:
        print(f"orodruin-benchmark: model_path={resolved_model_display}", file=sys.stderr)

    results: list[TaskResult] = []
    for task in tasks:
        config = RunConfig(
            backend=backend,
            model_path=effective_hf_model if backend == "hf" else args.model,
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
            run_id=run_id,
            run_label=args.run_label,
        )
        try:
            results.append(run_task(task, config))
        except (FileNotFoundError, ValueError) as exc:
            print(f"orodruin-benchmark: task {task.id} skipped: {exc}", file=sys.stderr)

    manifest = {
        "run_id": run_id,
        "experiment_dir": str(exp_root),
        "out_root": str(out_dir.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fp_payload,
        "fingerprint_digest": fingerprint_hash(fp_payload),
        "tasks": [
            {
                "task_id": r.task_id,
                "accuracy": r.accuracy,
                "total": r.total,
                "correct": r.correct,
                "elapsed_s": r.elapsed_s,
                "summary_path": str(r.summary_path.resolve()),
                "predictions_path": str(r.predictions_path.resolve()),
            }
            for r in results
        ],
    }
    summary_manifest_path = exp_root / "experiment_summary.json"
    summary_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"orodruin-benchmark: wrote {summary_manifest_path}", file=sys.stderr)

    _print_summary_table(results)


if __name__ == "__main__":
    main()
