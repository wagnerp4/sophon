from __future__ import annotations

from pathlib import Path

from training.finetune.datasets.load import load_registered_dataset
from training.finetune.recipe import FinetuneRecipe


def load_finetune_dataset(
    dataset_id: str,
    recipe: FinetuneRecipe,
    *,
    tokenizer: object | None = None,
    project_root: Path | None = None,
    split: str | None = None,
) -> object:
    return load_registered_dataset(
        dataset_id,
        recipe,
        tokenizer=tokenizer,
        project_root=project_root,
        split=split,
    )
