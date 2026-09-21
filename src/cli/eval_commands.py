from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from eval.runner import RunConfig, TaskResult, run_task
from eval.task_spec import find_task, list_task_ids, load_task
from utils.device.env_bootstrap import sophon_project_root


EmitFn = Callable[[str], None]


def _default_bench_data_dir() -> Path:
    env = os.environ.get("SOPHON_BENCH_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (sophon_project_root() / "data" / "benchmarks").resolve()


def _default_bench_out_dir() -> Path:
    env = os.environ.get("SOPHON_BENCH_OUT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (sophon_project_root() / "data" / "exps").resolve()


def format_eval_help() -> str:
    return "\n".join(
        [
            "eval suites:",
            "  /eval                     show suites + last run",
            "  /eval model [task] [N]    run model benchmark (default hellaswag)",
            "  /eval rag [N]             RAG ablation vs default LEANN index",
            "  /eval train               list indexed LoRA adapters",
            "  /rag-index [--rebuild]    rebuild vault+project corpus",
        ]
    )


def list_train_run_dirs(limit: int = 8) -> list[Path]:
    from training.common.index import load_adapter_index
    from utils.device.env_bootstrap import sophon_project_root

    entries = load_adapter_index(sophon_project_root()).entries
    if entries:
        return [Path(entry.path) for entry in reversed(entries)][: max(int(limit), 1)]
    root = sophon_project_root() / "data"
    candidates: list[Path] = []
    for rel in ("finetune", "exps", "training"):
        base = root / rel
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir():
                candidates.append(child)
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return candidates[: max(int(limit), 1)]


def format_train_index_lines(limit: int = 12) -> list[str]:
    from training.common.index import load_adapter_index
    from utils.device.env_bootstrap import sophon_project_root

    entries = load_adapter_index(sophon_project_root()).entries
    if not entries:
        return ["no adapters in data/training/adapters/index.json"]
    lines: list[str] = ["indexed adapters:"]
    for entry in entries[-max(int(limit), 1) :]:
        loss = f" loss={entry.last_loss:.4f}" if entry.last_loss is not None else ""
        lines.append(f"  {entry.name}: {entry.preset} / {entry.dataset} {entry.created}{loss}")
        lines.append(f"    {entry.path}")
    return lines


def run_model_eval(
    *,
    task_id: str = "hellaswag",
    limit: int | None = None,
    backend: str = "hf",
    model_path: str | None = None,
    preset_key: str | None = None,
    quantization: str = "none",
    server_model: str | None = None,
    ollama_model: str | None = None,
    on_emit: EmitFn | None = None,
) -> TaskResult:
    data_dir = _default_bench_data_dir()
    out_dir = _default_bench_out_dir()
    task = load_task(find_task(data_dir, task_id))
    root = sophon_project_root()
    resolved_model = model_path
    if preset_key and backend == "hf":
        from backend.hf.registry import resolve_preset_dir

        resolved_model = str(resolve_preset_dir(preset_key, root))
    model_name = server_model or ollama_model
    config = RunConfig(
        backend=backend,
        model_path=resolved_model if backend == "hf" else None,
        quantization=quantization,
        ollama_model=model_name if backend == "ollama" else None,
        server_model=model_name if backend in ("ollama", "lmstudio") else None,
        limit=limit,
        out_dir=out_dir,
        max_new_tokens=task.defaults.max_new_tokens,
    )
    if on_emit:
        on_emit(f"eval model: task={task_id} limit={limit} backend={backend}")
    return run_task(task, config)


def run_rag_eval(
    *,
    limit: int | None = 20,
    backend: str = "hf",
    model_path: str | None = None,
    preset_key: str | None = None,
    quantization: str = "none",
    server_model: str | None = None,
    ollama_model: str | None = None,
    on_emit: EmitFn | None = None,
) -> TaskResult:
    from eval.rag_ablation import run_rag_ablation

    data_dir = _default_bench_data_dir()
    out_dir = _default_bench_out_dir()
    task = load_task(find_task(data_dir, "rag_ablation"))
    root = sophon_project_root()
    resolved_model = model_path
    if preset_key and backend == "hf":
        from backend.hf.registry import resolve_preset_dir

        resolved_model = str(resolve_preset_dir(preset_key, root))
    model_name = server_model or ollama_model
    config = RunConfig(
        backend=backend,
        model_path=resolved_model if backend == "hf" else None,
        quantization=quantization,
        ollama_model=model_name if backend == "ollama" else None,
        server_model=model_name if backend in ("ollama", "lmstudio") else None,
        limit=limit,
        out_dir=out_dir,
        max_new_tokens=task.defaults.max_new_tokens,
    )
    return run_rag_ablation(task, config, on_emit=on_emit)


def summarize_task_result(result: TaskResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "total": result.total,
        "correct": result.correct,
        "accuracy": result.accuracy,
        "elapsed_s": result.elapsed_s,
        "summary_path": str(result.summary_path),
        "predictions_path": str(result.predictions_path),
    }


def available_model_tasks() -> list[str]:
    return list_task_ids(_default_bench_data_dir())
