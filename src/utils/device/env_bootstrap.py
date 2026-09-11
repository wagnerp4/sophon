from __future__ import annotations

import os
from pathlib import Path

DOTENV_LOAD_PATH: Path | None = None


def orodruin_project_root() -> Path:
    for anc in Path(__file__).resolve().parents[:10]:
        if (anc / "pyproject.toml").is_file():
            return anc
    return Path.cwd().resolve()


def orodruin_data_dir() -> Path:
    return orodruin_project_root() / "data"


def orodruin_chat_logs_dir() -> Path:
    from utils.device.platform import is_wsl, is_windows_mount_path

    root = orodruin_project_root()
    if is_wsl() and is_windows_mount_path(root):
        target = Path.home() / ".cache" / "orodruin" / "chat_logs"
    else:
        target = orodruin_data_dir() / "chat_logs"
    target.mkdir(parents=True, exist_ok=True)
    return target


def orodruin_assets_dir() -> Path:
    return orodruin_data_dir() / "assets"


def _dotenv_path() -> Path | None:
    explicit = os.environ.get("ORODRUIN_ENV_FILE", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None

    root_env = orodruin_project_root() / ".env"
    if root_env.is_file():
        return root_env.resolve()

    here = Path.cwd().resolve()
    for d in [here, *here.parents][:14]:
        for rel in (Path(".env"), Path("orodruin") / ".env"):
            cand = (d / rel).resolve()
            if cand.is_file():
                return cand
    return None


def load_orodruin_dotenv() -> None:
    global DOTENV_LOAD_PATH

    DOTENV_LOAD_PATH = None
    if os.environ.get("ORODRUIN_SKIP_DOTENV", "").strip().lower() in ("1", "true", "yes"):
        return
    from dotenv import load_dotenv

    path = _dotenv_path()
    if path is not None:
        load_dotenv(path, override=True, encoding="utf-8-sig")
        DOTENV_LOAD_PATH = path
