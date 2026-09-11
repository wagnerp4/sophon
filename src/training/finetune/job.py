from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch

from backend.hf.registry import model_dir_has_complete_weights, resolve_preset_dir
from training.common.paths import adapter_output_dir, new_run_id
from training.common.types import LogCallback, ProgressCallback, RunMeta
from training.finetune.backends.factory import load_finetune_backend, resolve_finetune_backend_id
from training.finetune.datasets.gsm8k import load_finetune_dataset
from training.finetune.datasets.registry import FINETUNE_DATASET_PRESETS, match_dataset_preset
from training.finetune.recipe import FinetuneRecipe, apply_recipe_overrides, parse_recipe_overrides
from training.finetune.smoke_eval import run_smoke_eval, write_smoke_eval


@dataclass(frozen=True)
class FinetuneJobResult:
    run_dir: Path
    adapter_dir: Path
    meta: RunMeta
    backend_id: str


def _cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_finetune_job(
    preset_key: str,
    dataset_id: str,
    *,
    project_root: Path | None = None,
    backend_id: str = "auto",
    recipe: FinetuneRecipe | None = None,
    recipe_override_tokens: list[str] | None = None,
    on_progress: ProgressCallback | None = None,
    on_log: LogCallback | None = None,
    run_smoke: bool = True,
) -> FinetuneJobResult:
    def log(msg: str) -> None:
        if on_log is not None:
            on_log(msg)

    matched_dataset = match_dataset_preset(dataset_id)
    if matched_dataset is None:
        raise ValueError(
            f"unknown finetune dataset {dataset_id!r}; "
            f"choose from {', '.join(sorted(FINETUNE_DATASET_PRESETS))}"
        )
    dataset_id = matched_dataset

    model_dir = resolve_preset_dir(preset_key, project_root)
    if not (model_dir / "config.json").is_file():
        raise ValueError(
            f"preset {preset_key!r} is not on disk at {model_dir}. "
            f"Run /model-download {preset_key} first."
        )
    if not model_dir_has_complete_weights(model_dir):
        raise ValueError(
            f"preset {preset_key!r} has incomplete weights at {model_dir}. "
            f"Run /model-download {preset_key} first."
        )

    base_recipe = recipe or FinetuneRecipe()
    overrides = parse_recipe_overrides(recipe_override_tokens or [])
    active_recipe = apply_recipe_overrides(base_recipe, overrides)

    resolved_backend = resolve_finetune_backend_id(backend_id)
    log(f"finetune backend: {resolved_backend}")
    backend = load_finetune_backend(resolved_backend)

    run_id = new_run_id()
    run_dir = adapter_output_dir(
        preset_key,
        dataset_id,
        project_root=project_root,
        run_id=run_id,
    )

    prepared = backend.prepare(model_dir, active_recipe, on_log=on_log)
    train_dataset = load_finetune_dataset(
        dataset_id,
        active_recipe,
        tokenizer=prepared.tokenizer,
    )

    smoke_before = None
    if run_smoke:
        log("running smoke eval (before finetune)")
        try:
            smoke_before = run_smoke_eval(prepared.model, prepared.tokenizer)
        except Exception as exc:
            log(f"(smoke eval before failed: {exc})")

    train_result = backend.train(
        prepared,
        train_dataset,
        active_recipe,
        run_dir,
        on_progress=on_progress,
        on_log=on_log,
    )
    adapter_dir = backend.save_adapter(prepared, run_dir)

    smoke_after = None
    if run_smoke:
        log("running smoke eval (after finetune)")
        try:
            smoke_after = run_smoke_eval(prepared.model, prepared.tokenizer)
        except Exception as exc:
            log(f"(smoke eval after failed: {exc})")

    smoke_payload = {"before": smoke_before, "after": smoke_after}
    write_smoke_eval(run_dir / "smoke_eval.json", smoke_payload)

    meta = RunMeta(
        preset_key=preset_key,
        dataset_id=dataset_id,
        backend_id=resolved_backend,
        base_model_path=str(model_dir.resolve()),
        adapter_dir=str(adapter_dir.resolve()),
        run_id=run_id,
        recipe={
            "max_seq_length": active_recipe.max_seq_length,
            "max_examples": active_recipe.max_examples,
            "num_train_epochs": active_recipe.num_train_epochs,
            "learning_rate": active_recipe.learning_rate,
            "per_device_train_batch_size": active_recipe.per_device_train_batch_size,
            "gradient_accumulation_steps": active_recipe.gradient_accumulation_steps,
            "lora_r": active_recipe.lora_r,
            "lora_alpha": active_recipe.lora_alpha,
        },
        train_metrics=train_result.metrics,
        smoke_eval=smoke_payload,
    )
    meta.write_json(run_dir / "run_meta.json")

    del prepared
    _cleanup_gpu()
    log(f"finetune complete -> {adapter_dir}")
    return FinetuneJobResult(
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        meta=meta,
        backend_id=resolved_backend,
    )
