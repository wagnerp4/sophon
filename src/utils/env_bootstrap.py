from __future__ import annotations

import os
from pathlib import Path

DOTENV_LOAD_PATH: Path | None = None


def lmwrap_project_root() -> Path:
    for anc in Path(__file__).resolve().parents[:10]:
        if (anc / "pyproject.toml").is_file():
            return anc
    return Path.cwd().resolve()


def _dotenv_path() -> Path | None:
    explicit = os.environ.get("LMWRAP_ENV_FILE", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None

    here = Path.cwd().resolve()
    for d in [here, *here.parents][:14]:
        for rel in (Path(".env"), Path("lmwrap") / ".env"):
            cand = (d / rel).resolve()
            if cand.is_file():
                return cand

    for anc in Path(__file__).resolve().parents[:10]:
        if (anc / "pyproject.toml").is_file():
            cand = anc / ".env"
            return cand if cand.is_file() else None
    return None


def load_lmwrap_dotenv() -> None:
    global DOTENV_LOAD_PATH

    DOTENV_LOAD_PATH = None
    if os.environ.get("LMWRAP_SKIP_DOTENV", "").strip().lower() in ("1", "true", "yes"):
        return
    from dotenv import load_dotenv

    path = _dotenv_path()
    if path is not None:
        load_dotenv(path, override=False, encoding="utf-8-sig")
        DOTENV_LOAD_PATH = path
