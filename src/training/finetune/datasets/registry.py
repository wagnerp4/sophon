from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinetuneDatasetPreset:
    hub_id: str
    split: str
    formatter_id: str
    max_examples: int
    description: str


FINETUNE_DATASET_PRESETS: dict[str, FinetuneDatasetPreset] = {
    "gsm8k_instructions": FinetuneDatasetPreset(
        hub_id="qwedsacf/grade-school-math-instructions",
        split="train",
        formatter_id="alpaca_instruction_response",
        max_examples=500,
        description="OpenAI grade-school-math instructions (Alpaca INSTRUCTION/RESPONSE). Notebook subset: 500 rows.",
    ),
}


def dataset_preset_keys_sorted() -> list[str]:
    return sorted(FINETUNE_DATASET_PRESETS.keys())


def match_dataset_preset(token: str) -> str | None:
    key = token.strip()
    if not key:
        return None
    if key in FINETUNE_DATASET_PRESETS:
        return key
    lower = key.lower()
    for preset_key in FINETUNE_DATASET_PRESETS:
        if preset_key.lower() == lower:
            return preset_key
    return None
