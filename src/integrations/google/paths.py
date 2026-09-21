from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from utils.device.env_bootstrap import sophon_data_dir, sophon_project_root
from utils.device.platform import is_wsl

_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def resolve_user_path(raw: str) -> Path | None:
    text = (raw or "").strip().strip("\"'")
    if not text:
        return None
    if is_wsl():
        match = _WIN_DRIVE_RE.match(text)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
    path = Path(text).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path


def google_data_dir() -> Path:
    path = sophon_data_dir() / "google"
    path.mkdir(parents=True, exist_ok=True)
    return path


def google_accounts_dir() -> Path:
    path = google_data_dir() / "accounts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def google_active_path() -> Path:
    return google_data_dir() / "active.json"


def client_secrets_path() -> Path | None:
    raw = os.environ.get("SOPHON_GOOGLE_CLIENT_SECRETS", "").strip()
    if raw:
        path = resolve_user_path(raw)
        if path is not None and path.is_file():
            return path
        mapped = _map_cross_os_project_path(raw)
        if mapped is not None and mapped.is_file():
            return mapped
    for candidate in _discover_client_secret_files():
        return candidate
    return None


def _map_cross_os_project_path(raw: str) -> Path | None:
    text = (raw or "").strip().strip("\"'")
    if not text:
        return None
    root = sophon_project_root()
    markers = (
        "/Computer Science/NLP/Personal/sophon/",
        "/NLP/Personal/sophon/",
        "\\Computer Science\\NLP\\Personal\\sophon\\",
        "\\NLP\\Personal\\sophon\\",
        "/sophon/",
        "\\sophon\\",
    )
    normalized = text.replace("\\", "/")
    for marker in markers:
        marker_norm = marker.replace("\\", "/")
        idx = normalized.lower().find(marker_norm.lower())
        if idx < 0:
            continue
        rel = normalized[idx + len(marker_norm) :]
        if not rel:
            continue
        candidate = root / Path(rel)
        try:
            return candidate.resolve()
        except OSError:
            return candidate
    name = Path(normalized).name
    if name.startswith("client_secret") and name.endswith(".json"):
        for folder in (google_data_dir(), google_accounts_dir(), root / "data" / "google"):
            cand = folder / name
            if cand.is_file():
                return cand
    return None


def _discover_client_secret_files() -> list[Path]:
    roots = [
        google_data_dir(),
        google_accounts_dir(),
        sophon_project_root() / "data" / "google",
        sophon_project_root() / "data" / "google" / "accounts",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            paths = sorted(root.glob("client_secret*.json"))
        except OSError:
            continue
        for path in paths:
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen or not path.is_file():
                continue
            if not _looks_like_oauth_client_file(path):
                continue
            seen.add(resolved)
            found.append(path)
    return found


def _looks_like_oauth_client_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")[:4000]
    except OSError:
        return False
    return '"installed"' in text or '"web"' in text


def account_token_path(email: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._@+-]+", "_", (email or "").strip().lower())
    if not safe:
        safe = "account"
    return google_accounts_dir() / f"{safe}.json"


def open_path_exists(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.exists()
    except OSError:
        return False


def is_windows() -> bool:
    return sys.platform == "win32"
