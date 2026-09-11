from __future__ import annotations

from training.finetune.datasets.formatters import format_alpaca_instruction_response
from training.finetune.datasets.registry import FINETUNE_DATASET_PRESETS, FinetuneDatasetPreset
from training.finetune.recipe import FinetuneRecipe


def _require_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "finetune dependencies missing. Run: uv sync --extra finetune"
        ) from exc
    return datasets


def load_finetune_dataset(
    dataset_id: str,
    recipe: FinetuneRecipe,
    *,
    tokenizer: object | None = None,
) -> object:
    preset = FINETUNE_DATASET_PRESETS.get(dataset_id)
    if preset is None:
        raise ValueError(f"unknown finetune dataset preset: {dataset_id!r}")

    if preset.formatter_id == "alpaca_instruction_response":
        return _load_gsm8k_instructions(preset, recipe, tokenizer=tokenizer)
    raise ValueError(f"unsupported formatter_id: {preset.formatter_id!r}")


def _load_gsm8k_instructions(
    preset: FinetuneDatasetPreset,
    recipe: FinetuneRecipe,
    *,
    tokenizer: object | None = None,
) -> object:
    _require_datasets()
    from datasets import load_dataset

    dataset = load_dataset(preset.hub_id, split=preset.split)
    max_examples = recipe.max_examples if recipe.max_examples > 0 else preset.max_examples
    if max_examples > 0 and len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))

    eos_token = ""
    if tokenizer is not None:
        eos = getattr(tokenizer, "eos_token", None)
        if isinstance(eos, str):
            eos_token = eos

    def formatting_prompts(examples: dict) -> dict[str, list[str]]:
        instructions = examples["INSTRUCTION"]
        responses = examples["RESPONSE"]
        texts: list[str] = []
        for instruction, response in zip(instructions, responses):
            texts.append(
                format_alpaca_instruction_response(
                    str(instruction),
                    str(response),
                    eos_token=eos_token,
                )
            )
        return {"text": texts}

    return dataset.map(formatting_prompts, batched=True)
