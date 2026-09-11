from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from training.common.types import LogCallback, ProgressCallback
from training.finetune.recipe import FinetuneRecipe


@dataclass
class PreparedModel:
    model: object
    tokenizer: object
    backend_id: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    metrics: dict[str, Any]
    adapter_dir: Path


class FinetuneBackend(Protocol):
    backend_id: str

    def prepare(
        self,
        base_model_path: Path,
        recipe: FinetuneRecipe,
        *,
        on_log: LogCallback | None = None,
    ) -> PreparedModel: ...

    def train(
        self,
        prepared: PreparedModel,
        train_dataset: object,
        recipe: FinetuneRecipe,
        output_dir: Path,
        *,
        on_progress: ProgressCallback | None = None,
        on_log: LogCallback | None = None,
    ) -> TrainResult: ...

    def save_adapter(self, prepared: PreparedModel, adapter_dir: Path) -> Path: ...
