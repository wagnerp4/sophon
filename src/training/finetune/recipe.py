from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LORA_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class FinetuneRecipe:
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_MODULES
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    logging_steps: int = 10
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 1
    max_examples: int = 500
    optim: str = "adamw_8bit"
    dataset: str = "gsm8k_instructions"
    preset: str | None = None
    output_name: str | None = None
    eval_split: str | None = None
    eval_steps: int | None = None
    report_curve: bool = True
    continue_from: str | None = None
    allow_large: bool = False
    oom_backoff: bool = False
    stages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    stage_id: str | None = None
    parent_adapter: str | None = None
    recipe_path: str | None = None


_BOOL_FIELDS = frozenset({"load_in_4bit", "report_curve", "allow_large", "oom_backoff"})
_INT_FIELDS = frozenset(
    {
        "max_seq_length",
        "lora_r",
        "lora_alpha",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_steps",
        "num_train_epochs",
        "logging_steps",
        "seed",
        "max_examples",
        "eval_steps",
    }
)
_FLOAT_FIELDS = frozenset({"lora_dropout", "learning_rate", "weight_decay"})
_STR_FIELDS = frozenset(
    {
        "lr_scheduler_type",
        "optim",
        "dataset",
        "preset",
        "output_name",
        "eval_split",
        "continue_from",
        "stage_id",
        "parent_adapter",
        "recipe_path",
    }
)
_ALIAS_TO_FIELD = {
    "epochs": "num_train_epochs",
    "lr": "learning_rate",
    "batch_size": "per_device_train_batch_size",
}


def config_finetune_dir(project_root: Path) -> Path:
    return project_root / "config" / "finetune"


def recipe_to_mapping(recipe: FinetuneRecipe) -> dict[str, Any]:
    payload = asdict(recipe)
    payload["lora_target_modules"] = list(recipe.lora_target_modules)
    payload["stages"] = [dict(stage) for stage in recipe.stages]
    return payload


def _parse_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected bool, got {raw!r}")


def _parse_optional_str(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _parse_modules(raw: object) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_LORA_MODULES
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",")]
        return tuple(item for item in parts if item)
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    raise ValueError(f"lora_target_modules must be a list or comma string, got {type(raw).__name__}")


def _parse_stages(raw: object) -> tuple[dict[str, Any], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("stages must be a list of mappings")
    stages: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each stage must be a mapping")
        stages.append(dict(item))
    return tuple(stages)


def mapping_to_recipe(raw: dict[str, Any], *, base: FinetuneRecipe | None = None) -> FinetuneRecipe:
    recipe = base or FinetuneRecipe()
    updates: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in raw.items():
        field_name = _ALIAS_TO_FIELD.get(key, key)
        if field_name == "id":
            field_name = "stage_id"
        if field_name not in FinetuneRecipe.__dataclass_fields__:
            unknown.append(str(key))
            continue
        if field_name == "lora_target_modules":
            updates[field_name] = _parse_modules(value)
        elif field_name == "stages":
            updates[field_name] = _parse_stages(value)
        elif field_name in _BOOL_FIELDS:
            updates[field_name] = _parse_bool(value)
        elif field_name in _INT_FIELDS:
            if value is None:
                updates[field_name] = None
            else:
                updates[field_name] = int(value)
        elif field_name in _FLOAT_FIELDS:
            updates[field_name] = float(value)
        elif field_name in _STR_FIELDS:
            updates[field_name] = _parse_optional_str(value) if field_name != "dataset" else (
                str(value).strip() if value is not None else recipe.dataset
            )
        else:
            updates[field_name] = value
    if unknown:
        raise ValueError(f"unknown recipe keys: {', '.join(unknown)}")
    if not updates:
        return recipe
    return replace(recipe, **updates)


def parse_recipe_overrides(tokens: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    unknown: list[str] = []
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"recipe override must be KEY=VAL, got {token!r}")
        key, _, raw_val = token.partition("=")
        key = key.strip()
        raw_val = raw_val.strip()
        if not key:
            raise ValueError(f"empty recipe override key in {token!r}")
        lower = key.lower()
        if lower in _ALIAS_TO_FIELD:
            field_name = _ALIAS_TO_FIELD[lower]
        elif lower in FinetuneRecipe.__dataclass_fields__:
            field_name = lower
        elif key in FinetuneRecipe.__dataclass_fields__:
            field_name = key
        else:
            unknown.append(key)
            continue
        out[field_name] = raw_val
    if unknown:
        raise ValueError(f"unknown recipe keys: {', '.join(unknown)}")
    return out


def apply_recipe_overrides(recipe: FinetuneRecipe, overrides: dict[str, object]) -> FinetuneRecipe:
    if not overrides:
        return recipe
    return mapping_to_recipe(dict(overrides), base=recipe)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"recipe YAML must be a mapping: {path}")
    return dict(raw)


def _maybe_yaml(path: Path, recipe: FinetuneRecipe) -> FinetuneRecipe:
    if not path.is_file():
        return recipe
    return mapping_to_recipe(_load_yaml_mapping(path), base=recipe)


def resolve_finetune_recipe(
    *,
    project_root: Path,
    dataset_id: str | None = None,
    preset_key: str | None = None,
    recipe_path: str | Path | None = None,
    override_tokens: list[str] | None = None,
) -> FinetuneRecipe:
    root = config_finetune_dir(project_root)
    recipe = FinetuneRecipe()
    recipe = _maybe_yaml(root / "default.yaml", recipe)
    if dataset_id:
        recipe = _maybe_yaml(root / "datasets" / f"{dataset_id}.yaml", recipe)
        recipe = replace(recipe, dataset=dataset_id)
    if preset_key:
        recipe = _maybe_yaml(root / f"{preset_key}.yaml", recipe)
        recipe = replace(recipe, preset=preset_key)
    loaded_path: Path | None = None
    if recipe_path:
        loaded_path = Path(recipe_path).expanduser()
        if not loaded_path.is_absolute():
            candidate = (project_root / loaded_path).resolve()
            if candidate.is_file():
                loaded_path = candidate
            else:
                loaded_path = loaded_path.resolve()
        if not loaded_path.is_file():
            raise FileNotFoundError(f"recipe file not found: {loaded_path}")
        recipe = mapping_to_recipe(_load_yaml_mapping(loaded_path), base=recipe)
        recipe = replace(recipe, recipe_path=str(loaded_path))
        if recipe.dataset:
            recipe = _maybe_yaml(root / "datasets" / f"{recipe.dataset}.yaml", recipe)
        if recipe.preset:
            recipe = _maybe_yaml(root / f"{recipe.preset}.yaml", recipe)
            recipe = mapping_to_recipe(_load_yaml_mapping(loaded_path), base=recipe)
            recipe = replace(recipe, recipe_path=str(loaded_path))
    if dataset_id:
        recipe = replace(recipe, dataset=dataset_id)
    if preset_key:
        recipe = replace(recipe, preset=preset_key)
    if override_tokens:
        recipe = apply_recipe_overrides(recipe, parse_recipe_overrides(override_tokens))
    return recipe


def merge_stage_recipe(parent: FinetuneRecipe, stage: dict[str, Any]) -> FinetuneRecipe:
    payload = dict(stage)
    stage_id = payload.pop("id", None)
    if stage_id is not None:
        payload["stage_id"] = str(stage_id)
    payload.pop("stages", None)
    child = mapping_to_recipe(payload, base=replace(parent, stages=()))
    return child


def write_resolved_recipe(path: Path, recipe: FinetuneRecipe) -> None:
    path.write_text(yaml.safe_dump(recipe_to_mapping(recipe), sort_keys=False), encoding="utf-8")
