from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinetuneDatasetPreset:
    formatter_id: str
    max_examples: int
    description: str
    hub_id: str | None = None
    split: str = "train"
    local_kind: str | None = None
    instruction_field: str = "INSTRUCTION"
    response_field: str = "RESPONSE"


FINETUNE_DATASET_PRESETS: dict[str, FinetuneDatasetPreset] = {
    "gsm8k_instructions": FinetuneDatasetPreset(
        formatter_id="alpaca_instruction_response",
        max_examples=500,
        description="OpenAI grade-school-math instructions (Alpaca INSTRUCTION/RESPONSE).",
        hub_id="qwedsacf/grade-school-math-instructions",
        split="train",
        instruction_field="INSTRUCTION",
        response_field="RESPONSE",
    ),
    "hellaswag": FinetuneDatasetPreset(
        formatter_id="hellaswag_ending",
        max_examples=500,
        description="HellaSwag commonsense completion SFT from local jsonl (activity+ctx -> gold ending).",
        split="train",
        local_kind="hellaswag_jsonl",
    ),
    "mmlu": FinetuneDatasetPreset(
        formatter_id="mmlu_choice",
        max_examples=500,
        description="MMLU multiple-choice SFT from local CSV (question+A-D -> letter and answer text).",
        split="train",
        local_kind="mmlu_csv",
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
