from __future__ import annotations

import gc
from dataclasses import dataclass, replace
from pathlib import Path

from backend.hf.registry import (
    model_dir_has_complete_weights,
    require_finetune_preset,
    resolve_preset_dir,
)
from training.common.curves import losses_from_metrics_jsonl
from training.common.index import (
    AdapterIndexEntry,
    register_adapter_entry,
    resolve_adapter_entry,
)
from training.common.paths import adapter_output_dir, new_run_id
from training.common.types import LogCallback, ProgressCallback, RunMeta
from training.finetune.backends.factory import load_finetune_backend, resolve_finetune_backend_id
from training.finetune.datasets.load import load_registered_dataset, maybe_split_eval
from training.finetune.datasets.registry import FINETUNE_DATASET_PRESETS, match_dataset_preset
from training.finetune.recipe import (
    FinetuneRecipe,
    merge_stage_recipe,
    recipe_to_mapping,
    resolve_finetune_recipe,
    write_resolved_recipe,
)
from training.finetune.runtime import (
    OOM_HINT,
    is_cuda_oom,
    oom_backoff_recipe,
    preflight_finetune,
)
from training.finetune.smoke_eval import run_smoke_eval, write_smoke_eval


@dataclass(frozen=True)
class FinetuneJobResult:
    run_dir: Path
    adapter_dir: Path
    meta: RunMeta
    backend_id: str
    adapter_name: str | None = None
    stage_results: tuple["FinetuneJobResult", ...] = ()


def _cleanup_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_continue_from(token: str | None, project_root: Path | None) -> Path | None:
    if not token:
        return None
    entry = resolve_adapter_entry(token, project_root=project_root)
    if entry is None:
        raise ValueError(f"continue_from adapter not found: {token!r}")
    path = Path(entry.path)
    if not path.is_dir():
        raise ValueError(f"continue_from path is not a directory: {path}")
    return path


def _adapter_display_name(recipe: FinetuneRecipe, run_id: str, dataset_id: str) -> str:
    if recipe.output_name and recipe.stage_id:
        return f"{recipe.output_name}-{recipe.stage_id}"
    if recipe.output_name:
        return recipe.output_name
    if recipe.stage_id:
        return f"{dataset_id}-{recipe.stage_id}-{run_id}"
    return f"{dataset_id}-{run_id}"


def _last_loss_and_steps(run_dir: Path, metrics: dict) -> tuple[float | None, int | None]:
    losses = losses_from_metrics_jsonl(run_dir / "metrics.jsonl")
    last_loss = losses[-1] if losses else None
    steps = len(losses) if losses else None
    if isinstance(metrics.get("train_loss"), (int, float)) and last_loss is None:
        last_loss = float(metrics["train_loss"])
    return last_loss, steps


def _run_one_stage(
    recipe: FinetuneRecipe,
    *,
    project_root: Path | None,
    backend_id: str,
    on_progress: ProgressCallback | None,
    on_log: LogCallback | None,
    run_smoke: bool,
) -> FinetuneJobResult:
    def log(msg: str) -> None:
        if on_log is not None:
            on_log(msg)

    dataset_id = recipe.dataset
    matched_dataset = match_dataset_preset(dataset_id)
    if matched_dataset is None:
        raise ValueError(
            f"unknown finetune dataset {dataset_id!r}; "
            f"choose from {', '.join(sorted(FINETUNE_DATASET_PRESETS))}"
        )
    dataset_id = matched_dataset
    recipe = replace(recipe, dataset=dataset_id)

    preset_key = recipe.preset
    if not preset_key:
        raise ValueError("finetune recipe is missing preset")
    require_finetune_preset(preset_key, allow_large=recipe.allow_large)

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

    continue_dir = _resolve_continue_from(recipe.continue_from, project_root)
    if continue_dir is not None:
        recipe = replace(recipe, parent_adapter=str(continue_dir))
        adapter_cfg = continue_dir / "adapter_config.json"
        if adapter_cfg.is_file():
            import json

            try:
                cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cfg = {}
            existing_r = cfg.get("r")
            if isinstance(existing_r, int) and existing_r != recipe.lora_r:
                raise ValueError(
                    f"continue_from adapter r={existing_r} does not match recipe lora_r={recipe.lora_r}"
                )

    resolved_backend = resolve_finetune_backend_id(backend_id)
    log(f"finetune backend: {resolved_backend}")
    if recipe.stage_id:
        log(f"stage {recipe.stage_id} dataset={dataset_id}")
    for line in preflight_finetune(model_dir=model_dir, recipe=recipe):
        log(line)

    backend = load_finetune_backend(resolved_backend)
    run_id = new_run_id()
    run_dir = adapter_output_dir(
        preset_key,
        dataset_id,
        project_root=project_root,
        run_id=run_id,
    )
    write_resolved_recipe(run_dir / "recipe.resolved.yaml", recipe)

    def _train_with_recipe(active: FinetuneRecipe):
        prepared = backend.prepare(
            model_dir,
            active,
            on_log=on_log,
            adapter_path=continue_dir,
        )
        train_dataset = load_registered_dataset(
            dataset_id,
            active,
            tokenizer=prepared.tokenizer,
            project_root=project_root,
        )
        train_dataset, eval_dataset = maybe_split_eval(
            train_dataset,
            active,
            dataset_id=dataset_id,
            tokenizer=prepared.tokenizer,
            project_root=project_root,
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
            active,
            run_dir,
            eval_dataset=eval_dataset,
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
        return prepared, train_result, adapter_dir, smoke_before, smoke_after

    active_recipe = recipe
    prepared = None
    try:
        prepared, train_result, adapter_dir, smoke_before, smoke_after = _train_with_recipe(active_recipe)
    except Exception as exc:
        _cleanup_gpu()
        if not (is_cuda_oom(exc) and active_recipe.oom_backoff):
            if is_cuda_oom(exc):
                raise RuntimeError(OOM_HINT) from exc
            raise
        log(f"CUDA OOM; retrying once with {OOM_HINT}")
        active_recipe = oom_backoff_recipe(active_recipe)
        log(
            f"oom_backoff batch={active_recipe.per_device_train_batch_size} "
            f"accum={active_recipe.gradient_accumulation_steps}"
        )
        write_resolved_recipe(run_dir / "recipe.resolved.yaml", active_recipe)
        try:
            prepared, train_result, adapter_dir, smoke_before, smoke_after = _train_with_recipe(active_recipe)
        except Exception as retry_exc:
            _cleanup_gpu()
            if is_cuda_oom(retry_exc):
                raise RuntimeError(OOM_HINT) from retry_exc
            raise

    smoke_payload = {"before": smoke_before, "after": smoke_after}
    write_smoke_eval(run_dir / "smoke_eval.json", smoke_payload)
    last_loss, steps = _last_loss_and_steps(run_dir, train_result.metrics)
    adapter_name = _adapter_display_name(active_recipe, run_id, dataset_id)
    indexed = register_adapter_entry(
        AdapterIndexEntry(
            name=adapter_name,
            preset=preset_key,
            dataset=dataset_id,
            run_id=run_id,
            path=str(adapter_dir.resolve()),
            created=run_id,
            last_loss=last_loss,
            steps=steps,
            parent=active_recipe.parent_adapter,
            stage_id=active_recipe.stage_id,
        ),
        project_root=project_root,
    )
    meta = RunMeta(
        preset_key=preset_key,
        dataset_id=dataset_id,
        backend_id=resolved_backend,
        base_model_path=str(model_dir.resolve()),
        adapter_dir=str(adapter_dir.resolve()),
        run_id=run_id,
        recipe=recipe_to_mapping(active_recipe),
        train_metrics=train_result.metrics,
        smoke_eval=smoke_payload,
        parent_adapter=active_recipe.parent_adapter,
        stage_id=active_recipe.stage_id,
        adapter_name=indexed.name,
    )
    meta.write_json(run_dir / "run_meta.json")
    del prepared
    _cleanup_gpu()
    log(f"finetune complete -> {adapter_dir}")
    log(f"adapter name={indexed.name}; /adapter load {indexed.name}")
    # TODO: pin distillation/eval teachers via known_managed_teacher_ids from the versioned managed model index
    return FinetuneJobResult(
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        meta=meta,
        backend_id=resolved_backend,
        adapter_name=indexed.name,
    )


def run_finetune_job(
    preset_key: str | None,
    dataset_id: str | None,
    *,
    project_root: Path | None = None,
    backend_id: str = "auto",
    recipe: FinetuneRecipe | None = None,
    recipe_path: str | Path | None = None,
    recipe_override_tokens: list[str] | None = None,
    on_progress: ProgressCallback | None = None,
    on_log: LogCallback | None = None,
    run_smoke: bool = True,
) -> FinetuneJobResult:
    root = project_root
    if recipe is None:
        recipe = resolve_finetune_recipe(
            project_root=root or Path.cwd(),
            dataset_id=dataset_id,
            preset_key=preset_key,
            recipe_path=recipe_path,
            override_tokens=recipe_override_tokens,
        )
    else:
        updates: dict[str, object] = {}
        if dataset_id:
            updates["dataset"] = dataset_id
        if preset_key:
            updates["preset"] = preset_key
        if updates:
            recipe = replace(recipe, **updates)
        if recipe_override_tokens:
            from training.finetune.recipe import apply_recipe_overrides, parse_recipe_overrides

            recipe = apply_recipe_overrides(recipe, parse_recipe_overrides(recipe_override_tokens))

    if recipe.stages:
        results: list[FinetuneJobResult] = []
        parent_path: str | None = recipe.continue_from
        for index, stage in enumerate(recipe.stages):
            stage_recipe = merge_stage_recipe(recipe, stage)
            if index > 0:
                if stage_recipe.lora_r != recipe.lora_r:
                    raise ValueError("cannot change lora_r mid-pipeline (adapter rank must match)")
                parent_path = str(results[-1].adapter_dir)
            if parent_path:
                stage_recipe = replace(stage_recipe, continue_from=parent_path)
            result = _run_one_stage(
                stage_recipe,
                project_root=root,
                backend_id=backend_id,
                on_progress=on_progress,
                on_log=on_log,
                run_smoke=run_smoke and index == len(recipe.stages) - 1,
            )
            results.append(result)
            parent_path = str(result.adapter_dir)
        last = results[-1]
        return FinetuneJobResult(
            run_dir=last.run_dir,
            adapter_dir=last.adapter_dir,
            meta=last.meta,
            backend_id=last.backend_id,
            adapter_name=last.adapter_name,
            stage_results=tuple(results),
        )

    # TODO: Gemma 4 / Llama 4 / Qwen multimodal SFT needs architecture-specific trainers, not AutoModelForCausalLM.
    return _run_one_stage(
        recipe,
        project_root=root,
        backend_id=backend_id,
        on_progress=on_progress,
        on_log=on_log,
        run_smoke=run_smoke,
    )
