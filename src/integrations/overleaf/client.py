from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.device.env_bootstrap import sophon_data_dir

_GIT_HOST = "git.overleaf.com"
_MAX_BODY_CHARS = 12_000
_SECTION_RE = re.compile(
    r"^\\(section|subsection|subsubsection)\*?\{([^{}]*)\}",
    re.MULTILINE,
)
_TEXT_SUFFIXES = {
    ".tex",
    ".bib",
    ".sty",
    ".cls",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".bst",
}


@dataclass
class OverleafProject:
    project_id: str
    alias: str
    path: Path
    sync_error: str | None = None


@dataclass
class OverleafEntry:
    name: str
    relpath: str
    is_dir: bool


def overleaf_git_token() -> str | None:
    for key in ("SOPHON_OVERLEAF_GIT_TOKEN", "OVERLEAF_GIT_TOKEN"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    return None


def overleaf_data_dir() -> Path:
    path = sophon_data_dir() / "overleaf"
    path.mkdir(parents=True, exist_ok=True)
    return path


def overleaf_projects_dir() -> Path:
    path = overleaf_data_dir() / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def overleaf_aliases_path() -> Path:
    return overleaf_data_dir() / "projects.json"


def _load_aliases() -> dict[str, str]:
    path = overleaf_aliases_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for alias, project_id in raw.items():
        name = str(alias).strip()
        pid = str(project_id).strip()
        if name and pid:
            out[name] = pid
    return out


def configured_project_ids() -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        for part in raw.split(","):
            pid = part.strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)

    single = os.environ.get("SOPHON_OVERLEAF_PROJECT_ID", "").strip()
    if single:
        _add(single)
    multi = os.environ.get("SOPHON_OVERLEAF_PROJECT_IDS", "").strip()
    if multi:
        _add(multi)
    for project_id in _load_aliases().values():
        _add(project_id)
    return ids


def overleaf_available() -> bool:
    return overleaf_git_token() is not None and bool(configured_project_ids())


def overleaf_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_OVERLEAF_TOOLS", "0").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    return overleaf_available()


def project_local_path(project_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", project_id.strip())
    if not safe:
        safe = "project"
    return overleaf_projects_dir() / safe


def _alias_for(project_id: str) -> str:
    for alias, pid in _load_aliases().items():
        if pid == project_id:
            return alias
    return project_id


def _tokenless_url(project_id: str) -> str:
    return f"https://{_GIT_HOST}/{project_id}"


def _authed_url(project_id: str, token: str) -> str:
    return f"https://git:{token}@{_GIT_HOST}/{project_id}"


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        env=_git_env(),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _rewrite_origin(local: Path, project_id: str) -> None:
    result = _run_git(
        ["remote", "set-url", "origin", _tokenless_url(project_id)],
        cwd=local,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git remote set-url failed: {detail[:500]}")


def ensure_synced(project_id: str) -> Path:
    # TODO: write + commit/push back to Overleaf Git
    # TODO: web-session compile / PDF fetch
    pid = (project_id or "").strip()
    if not pid:
        raise RuntimeError("project_id is required")
    token = overleaf_git_token()
    if not token:
        raise RuntimeError(
            "Set SOPHON_OVERLEAF_GIT_TOKEN (Overleaf Account Settings → Git Integration)."
        )
    if pid not in configured_project_ids():
        raise RuntimeError(
            f"Unknown Overleaf project {pid!r}. "
            "Add it via SOPHON_OVERLEAF_PROJECT_ID / SOPHON_OVERLEAF_PROJECT_IDS "
            "or data/overleaf/projects.json."
        )
    local = project_local_path(pid)
    if (local / ".git").is_dir():
        result = _run_git(
            [
                "-c",
                f"http.extraHeader=Authorization: Basic {_basic_auth('git', token)}",
                "pull",
                "--ff-only",
            ],
            cwd=local,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"git pull failed for {pid}: {detail[:500]}")
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        raise RuntimeError(f"Clone path exists but is not a git repo: {local}")
    result = _run_git(
        ["clone", "--depth", "1", _authed_url(pid, token), str(local)],
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git clone failed for {pid}: {detail[:500]}")
    _rewrite_origin(local, pid)
    return local


def _basic_auth(user: str, password: str) -> str:
    raw = f"{user}:{password}".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def list_projects(*, sync: bool = True) -> list[OverleafProject]:
    out: list[OverleafProject] = []
    for project_id in configured_project_ids():
        path = project_local_path(project_id)
        sync_error: str | None = None
        if sync:
            try:
                path = ensure_synced(project_id)
            except Exception as exc:
                sync_error = str(exc)
        out.append(
            OverleafProject(
                project_id=project_id,
                alias=_alias_for(project_id),
                path=path,
                sync_error=sync_error,
            )
        )
    return out


def _resolve_under(project_root: Path, relpath: str) -> Path:
    cleaned = (relpath or "").replace("\\", "/").strip().lstrip("/")
    if cleaned in ("", "."):
        target = project_root
    else:
        parts = [part for part in cleaned.split("/") if part and part != "."]
        if any(part == ".." for part in parts):
            raise RuntimeError("path must stay inside the project")
        target = project_root.joinpath(*parts)
    try:
        resolved = target.resolve()
        root = project_root.resolve()
    except OSError as exc:
        raise RuntimeError(f"invalid path: {exc}") from exc
    if resolved != root and root not in resolved.parents:
        raise RuntimeError("path must stay inside the project")
    return resolved


def list_files(project_id: str, relpath: str = "") -> list[OverleafEntry]:
    root = ensure_synced(project_id)
    folder = _resolve_under(root, relpath)
    if not folder.is_dir():
        raise RuntimeError(f"not a directory: {relpath or '/'}")
    entries: list[OverleafEntry] = []
    try:
        children = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        raise RuntimeError(f"list failed: {exc}") from exc
    for child in children:
        if child.name == ".git":
            continue
        try:
            rel = child.relative_to(root).as_posix()
        except ValueError:
            continue
        entries.append(OverleafEntry(name=child.name, relpath=rel, is_dir=child.is_dir()))
    return entries


def read_file(project_id: str, relpath: str, *, truncate: bool = False) -> str:
    root = ensure_synced(project_id)
    path = _resolve_under(root, relpath)
    if not path.is_file():
        raise RuntimeError(f"not a file: {relpath}")
    suffix = path.suffix.lower()
    if suffix and suffix not in _TEXT_SUFFIXES and suffix not in {".tex", ".bib"}:
        if suffix in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
            raise RuntimeError(f"binary file ({suffix}); open it in the editor tree instead")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"read failed: {exc}") from exc
    if truncate:
        return _truncate(text)
    return text


def list_sections(project_id: str, relpath: str) -> list[dict[str, Any]]:
    text = read_file(project_id, relpath, truncate=False)
    rows: list[dict[str, Any]] = []
    for match in _SECTION_RE.finditer(text):
        kind = match.group(1)
        title = match.group(2).strip()
        line = text.count("\n", 0, match.start()) + 1
        level = {"section": 1, "subsection": 2, "subsubsection": 3}.get(kind, 1)
        rows.append({"level": level, "kind": kind, "title": title, "line": line})
    return rows


def format_tool_payload(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _truncate(text: str, limit: int = _MAX_BODY_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def project_file_path(project_id: str, relpath: str) -> Path:
    root = ensure_synced(project_id)
    return _resolve_under(root, relpath)


def ping() -> dict[str, Any]:
    token = overleaf_git_token()
    ids = configured_project_ids()
    return {
        "ok": bool(token) and bool(ids),
        "token": bool(token),
        "projects": ids,
        "aliases": _load_aliases(),
        "cache": str(overleaf_projects_dir()),
    }
