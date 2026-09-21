from __future__ import annotations

import datetime as _dt
from pathlib import Path

from utils.device.env_bootstrap import sophon_data_dir


def training_runs_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        root = (project_root / "data" / "training").resolve()
    else:
        root = sophon_data_dir() / "training"
    root.mkdir(parents=True, exist_ok=True)
    return root


def adapters_root(project_root: Path | None = None) -> Path:
    root = training_runs_root(project_root) / "adapters"
    root.mkdir(parents=True, exist_ok=True)
    return root


def checkpoints_root(project_root: Path | None = None) -> Path:
    root = training_runs_root(project_root) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def adapter_output_dir(
    preset_key: str,
    dataset_id: str,
    *,
    project_root: Path | None = None,
    run_id: str | None = None,
) -> Path:
    rid = run_id or new_run_id()
    out = adapters_root(project_root) / preset_key / dataset_id / rid
    out.mkdir(parents=True, exist_ok=True)
    return out
