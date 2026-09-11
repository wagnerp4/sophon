from __future__ import annotations

import re
import shlex
from pathlib import Path

_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".less",
    ".log",
    ".md",
    ".mjs",
    ".php",
    ".py",
    ".rs",
    ".rst",
    ".sass",
    ".scss",
    ".sh",
    ".sql",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

_URI_RE = re.compile(r"^(?:file:///)(.+)$", re.IGNORECASE)
_MAX_ATTACH_CHARS = 24_000
_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


def _strip_uri(token: str) -> str:
    raw = token.strip().strip("\"'")
    match = _URI_RE.match(raw)
    if match is None:
        return raw
    path = match.group(1)
    if re.match(r"^[A-Za-z]:/", path):
        return path.replace("/", "\\")
    if path.startswith("/") and len(path) > 2 and path[2] == ":":
        return path[1:].replace("/", "\\")
    return path


def _tokenize_paste(text: str) -> list[str]:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    try:
        parts = shlex.split(cleaned, posix=False)
    except ValueError:
        parts = []
    if parts:
        return [p for p in parts if p.strip()]
    return [line.strip() for line in cleaned.split("\n") if line.strip()]


def extract_dropped_paths(text: str) -> list[Path]:
    tokens = _tokenize_paste(text)
    if not tokens:
        return []
    found: list[Path] = []
    seen: set[str] = set()
    for token in tokens:
        candidate = Path(_strip_uri(token)).expanduser()
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        key = str(resolved).casefold()
        if key in seen:
            continue
        if not resolved.exists():
            continue
        seen.add(key)
        found.append(resolved)
    if not found:
        return []
    if len(found) != len(tokens):
        return []
    return found


def is_audio_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES


def split_audio_paths(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    audio: list[Path] = []
    other: list[Path] = []
    for path in paths:
        if is_audio_path(path):
            audio.append(path)
        else:
            other.append(path)
    return audio, other


def format_paths_for_prompt(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        text = str(path)
        if any(ch in text for ch in (" ", "\t")):
            chunks.append(f"\"{text}\"")
        else:
            chunks.append(text)
    return " ".join(chunks)


def describe_dropped_paths(paths: list[Path]) -> str:
    bits: list[str] = []
    for path in paths:
        kind = "dir" if path.is_dir() else "file"
        size = ""
        if path.is_file():
            try:
                nbytes = path.stat().st_size
                if nbytes < 1024:
                    size = f" · {nbytes} B"
                else:
                    size = f" · {nbytes / 1024:.1f} KiB"
            except OSError:
                size = ""
        bits.append(f"{path.name} ({kind}{size})")
    label = "file" if len(paths) == 1 else "files"
    return f"dropped {len(paths)} {label}: " + ", ".join(bits)


def _looks_like_text(path: Path) -> bool:
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return True
    try:
        sample = path.read_bytes()[:2048]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def build_message_with_attachments(
    user_text: str,
    paths: list[Path],
    *,
    max_chars: int = _MAX_ATTACH_CHARS,
) -> str:
    remaining = [path for path in paths if path.exists()]
    if not remaining:
        return user_text

    prompt = user_text.strip()
    path_blob = format_paths_for_prompt(remaining)
    if not prompt or prompt == path_blob:
        prompt = "Please review the attached file(s)."

    sections: list[str] = [prompt]
    budget = max(1024, int(max_chars))
    for path in remaining:
        if path.is_dir():
            sections.append(f"\n--- path: {path} (directory) ---")
            continue
        if not path.is_file():
            sections.append(f"\n--- path: {path} (missing) ---")
            continue
        if not _looks_like_text(path):
            sections.append(f"\n--- file: {path} (binary/unsupported, path only) ---")
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            sections.append(f"\n--- file: {path} (read error: {exc}) ---")
            continue
        truncated = False
        if len(body) > budget:
            body = body[:budget]
            truncated = True
        note = " truncated" if truncated else ""
        sections.append(f"\n--- file: {path}{note} ---\n{body}\n--- end ---")
        budget = max(0, budget - len(body))
        if budget <= 0:
            sections.append("\n(additional file contents omitted)")
            break
    return "\n".join(sections).strip()


def paths_still_in_text(text: str, paths: list[Path]) -> bool:
    if not paths:
        return False
    hay = text.replace("/", "\\")
    for path in paths:
        needle = str(path).replace("/", "\\")
        if needle not in hay and path.name not in text:
            return False
    return True
