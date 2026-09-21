from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from utils.device.env_bootstrap import sophon_project_root

DEFAULT_SSL4SED_ROOT = "/home/philipp/software/python/SSL4SED"
DEFAULT_WSL_DISTRO = "Debian"


@dataclass(frozen=True)
class TrainerConfig:
    ssl4sed_root: str
    wsl_distro: str
    default_ssl4sed_target: str


def _read_simple_mapping(path: Path) -> dict:
    out: dict = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def _read_yaml_mapping(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return _read_simple_mapping(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return raw


def load_trainer_config(project_root: Path | None = None) -> TrainerConfig:
    root = project_root or sophon_project_root()
    mapping = _read_yaml_mapping(root / "config" / "trainer.yaml")
    ssl4sed_root = str(
        os.environ.get("SOPHON_SSL4SED_ROOT", "").strip()
        or mapping.get("ssl4sed_root")
        or DEFAULT_SSL4SED_ROOT
    )
    wsl_distro = str(
        os.environ.get("SOPHON_WSL_DISTRO", "").strip()
        or mapping.get("wsl_distro")
        or DEFAULT_WSL_DISTRO
    )
    default_target = str(
        os.environ.get("SOPHON_SSL4SED_TARGET", "").strip()
        or mapping.get("default_ssl4sed_target")
        or ""
    )
    return TrainerConfig(
        ssl4sed_root=ssl4sed_root.rstrip("/"),
        wsl_distro=wsl_distro,
        default_ssl4sed_target=default_target.strip(),
    )
