from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml

from utils.device.env_bootstrap import sophon_project_root

HIGHLIGHTS_FILENAME = "path_highlights.yaml"

_DEFAULT_COLOR = "#e5e5e5"
_DEFAULT_ROOTS: dict[str, str] = {
    "vault": "#7c3aed",
    "obsidian": "#7c3aed",
    "youtube": "#dc2626",
    "drive": "#3b82f6",
    "overleaf": "#16a34a",
    "zotero": "#cc3300",
    "project": _DEFAULT_COLOR,
    "disk": _DEFAULT_COLOR,
    "bookmarks": _DEFAULT_COLOR,
}
_DEFAULT_SEGMENTS: dict[str, str] = {
    "personal": "vault",
    "kb": "vault",
    "repos": "vault",
    "storage": "vault",
    "templates": "vault",
    "youtube": "youtube",
    "youtu.be": "youtube",
}

_TEMPLATE = """# Background chips for `path` / `url` spans in assistant replies.
# Unknown paths use default. Add roots or segments over time.

default: "#e5e5e5"
roots:
  vault: "#7c3aed"
  obsidian: "#7c3aed"
  youtube: "#dc2626"
  drive: "#3b82f6"
  overleaf: "#16a34a"
  zotero: "#cc3300"
  project: "#e5e5e5"
  disk: "#e5e5e5"
  bookmarks: "#e5e5e5"
segments:
  Personal: vault
  KB: vault
  Repos: vault
  Storage: vault
  Templates: vault
"""

_cached: tuple[float, Path, dict[str, object]] | None = None


def highlights_path(root: Path | None = None) -> Path:
    return (root or sophon_project_root()) / ".sophon" / HIGHLIGHTS_FILENAME


def ensure_highlights_file(root: Path | None = None) -> Path:
    path = highlights_path(root)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE, encoding="utf-8")
    return path


def _load_raw(root: Path | None = None) -> dict[str, object]:
    global _cached
    path = ensure_highlights_file(root)
    mtime = 0.0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _cached is not None and _cached[0] == mtime and _cached[1] == path:
        return _cached[2]
    data: dict[str, object] = {
        "default": _DEFAULT_COLOR,
        "roots": dict(_DEFAULT_ROOTS),
        "segments": dict(_DEFAULT_SEGMENTS),
    }
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        loaded = None
    if isinstance(loaded, dict):
        default = loaded.get("default")
        if isinstance(default, str) and default.strip():
            data["default"] = default.strip()
        roots = loaded.get("roots")
        if isinstance(roots, dict):
            merged_roots = dict(_DEFAULT_ROOTS)
            for key, value in roots.items():
                if isinstance(value, str) and str(key).strip():
                    merged_roots[str(key).strip().lower()] = value.strip()
            data["roots"] = merged_roots
        segments = loaded.get("segments")
        if isinstance(segments, dict):
            merged_seg = dict(_DEFAULT_SEGMENTS)
            for key, value in segments.items():
                if str(key).strip() and isinstance(value, str) and value.strip():
                    merged_seg[str(key).strip().lower()] = value.strip().lower()
            data["segments"] = merged_seg
    _cached = (mtime, path, data)
    return data


def color_for_span(text: str, root: Path | None = None) -> str:
    cfg = _load_raw(root)
    default = str(cfg.get("default") or _DEFAULT_COLOR)
    roots = cfg.get("roots") if isinstance(cfg.get("roots"), dict) else _DEFAULT_ROOTS
    segments = cfg.get("segments") if isinstance(cfg.get("segments"), dict) else _DEFAULT_SEGMENTS
    raw = (text or "").strip()
    if not raw:
        return default
    host = ""
    lowered = raw.lower()
    if "://" in raw or lowered.startswith("www."):
        parsed = urlparse(raw if "://" in raw else "https://" + raw)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if "youtube" in host or host.endswith("youtu.be"):
            return str(roots.get("youtube") or default)
        if "drive.google" in host:
            return str(roots.get("drive") or default)
        if "overleaf" in host:
            return str(roots.get("overleaf") or default)
        if "zotero" in host:
            return str(roots.get("zotero") or default)
    first = raw.replace("\\", "/").split("/")[0].strip()
    if not first:
        return default
    key = first.lower()
    mapped = segments.get(key) if isinstance(segments, dict) else None
    if isinstance(mapped, str) and mapped:
        return str(roots.get(mapped) or default) if isinstance(roots, dict) else default
    if isinstance(roots, dict) and key in roots:
        return str(roots.get(key) or default)
    return default


def chip_markup(inner_escaped: str, color: str) -> str:
    tone = (color or _DEFAULT_COLOR).strip() or _DEFAULT_COLOR
    return "[black on " + tone + "]" + inner_escaped + "[/]"
