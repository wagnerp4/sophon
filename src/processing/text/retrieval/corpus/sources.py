from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_VAULT_SUFFIXES = {".md", ".markdown", ".txt"}
_PROJECT_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".rst",
    ".tex",
    ".sh",
    ".ps1",
    ".css",
    ".html",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
}
_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".eggs",
    "models",
    ".obsidian",
    ".trash",
    "trash",
}
_MAX_FILE_BYTES = 1_500_000


@dataclass(frozen=True)
class CorpusFile:
    path: Path
    source: str
    rel: str


def read_vault_path_from_env() -> Path | None:
    raw = os.environ.get("SOPHON_VAULT_PATH", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path


def _should_skip_dir(name: str) -> bool:
    lower = name.lower()
    if lower in _SKIP_DIR_NAMES:
        return True
    if name.startswith(".") and lower not in {".github"}:
        return True
    return False


def _iter_files(root: Path, *, suffixes: set[str], source: str) -> list[CorpusFile]:
    if not root.is_dir():
        return []
    out: list[CorpusFile] = []
    root_resolved = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root_resolved):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            if path.suffix.lower() not in suffixes:
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            try:
                rel = str(path.relative_to(root_resolved)).replace("\\", "/")
            except ValueError:
                rel = path.name
            out.append(CorpusFile(path=path, source=source, rel=rel))
    out.sort(key=lambda item: (item.source, item.rel))
    return out


def collect_vault_files(vault_root: Path) -> list[CorpusFile]:
    return _iter_files(vault_root, suffixes=_VAULT_SUFFIXES, source="obsidian")


def collect_project_files(project_root: Path) -> list[CorpusFile]:
    return _iter_files(project_root, suffixes=_PROJECT_SUFFIXES, source="project")


def collect_corpus_files(
    *,
    vault_root: Path | None = None,
    project_root: Path | None = None,
    include_project: bool = True,
) -> list[CorpusFile]:
    files: list[CorpusFile] = []
    vault = vault_root if vault_root is not None else read_vault_path_from_env()
    if vault is not None:
        files.extend(collect_vault_files(vault))
    if include_project and project_root is not None:
        files.extend(collect_project_files(project_root))
    return files
