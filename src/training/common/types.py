from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class RunMeta:
    preset_key: str
    dataset_id: str
    backend_id: str
    base_model_path: str
    adapter_dir: str
    run_id: str
    recipe: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    smoke_eval: dict[str, Any] = field(default_factory=dict)

    def write_json(self, path: Path) -> None:
        import json

        payload = {
            "preset_key": self.preset_key,
            "dataset_id": self.dataset_id,
            "backend_id": self.backend_id,
            "base_model_path": self.base_model_path,
            "adapter_dir": self.adapter_dir,
            "run_id": self.run_id,
            "recipe": self.recipe,
            "train_metrics": self.train_metrics,
            "smoke_eval": self.smoke_eval,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> RunMeta:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            preset_key=str(raw["preset_key"]),
            dataset_id=str(raw["dataset_id"]),
            backend_id=str(raw["backend_id"]),
            base_model_path=str(raw["base_model_path"]),
            adapter_dir=str(raw["adapter_dir"]),
            run_id=str(raw["run_id"]),
            recipe=dict(raw.get("recipe") or {}),
            train_metrics=dict(raw.get("train_metrics") or {}),
            smoke_eval=dict(raw.get("smoke_eval") or {}),
        )
