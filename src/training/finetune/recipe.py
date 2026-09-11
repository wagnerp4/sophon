from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FinetuneRecipe:
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
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


_RECIPE_OVERRIDE_KEYS = {
    "max_seq_length": int,
    "max_examples": int,
    "epochs": ("num_train_epochs", int),
    "num_train_epochs": int,
    "lr": ("learning_rate", float),
    "learning_rate": float,
    "batch_size": ("per_device_train_batch_size", int),
    "per_device_train_batch_size": int,
    "gradient_accumulation_steps": int,
    "lora_r": int,
    "lora_alpha": int,
    "seed": int,
}


def parse_recipe_overrides(tokens: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, _, raw_val = token.partition("=")
        key = key.strip().lower()
        raw_val = raw_val.strip()
        if not key or not raw_val:
            continue
        spec = _RECIPE_OVERRIDE_KEYS.get(key)
        if spec is None:
            continue
        if isinstance(spec, tuple):
            field_name, caster = spec
        else:
            field_name, caster = key, spec
        try:
            out[field_name] = caster(raw_val)
        except (TypeError, ValueError):
            continue
    return out


def apply_recipe_overrides(recipe: FinetuneRecipe, overrides: dict[str, object]) -> FinetuneRecipe:
    if not overrides:
        return recipe
    return replace(recipe, **overrides)
