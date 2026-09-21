from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from processing.text.completion.types import CompletionCandidate, CompletionQuery

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOTTED_PREFIX_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*\Z")
_MIN_PREFIX_LEN = 2

_DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "sophon.egg-info",
        ".tox",
    }
)
_DEFAULT_SCAN_ROOTS = ("src", "scripts")
_DEFAULT_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".md",
        ".markdown",
        ".svg",
        ".stl",
        ".csv",
        ".tsv",
        ".tab",
        ".jsonl",
        ".xlsx",
        ".ipynb",
    }
)


def tokenize_identifiers(text: str) -> list[str]:
    return _IDENT_RE.findall(text)


def extract_prefix_at_cursor(document_text: str, cursor_offset: int) -> tuple[str, str]:
    if cursor_offset < 0:
        cursor_offset = 0
    if cursor_offset > len(document_text):
        cursor_offset = len(document_text)
    before = document_text[:cursor_offset]
    after = document_text[cursor_offset:]
    match = _DOTTED_PREFIX_RE.search(before)
    if match is None:
        return "", after
    if after and (after[0].isalnum() or after[0] == "_"):
        return "", after
    return match.group(0), after


def _best_token(prefix: str, counts: Counter[str]) -> str | None:
    if len(prefix) < _MIN_PREFIX_LEN:
        return None
    best: str | None = None
    best_count = -1
    for token, count in counts.items():
        if token == prefix:
            continue
        if not token.startswith(prefix):
            continue
        if count > best_count or (count == best_count and best is not None and token < best):
            best = token
            best_count = count
        elif best is None:
            best = token
            best_count = count
    return best


class PrefixCompleter:
    def __init__(
        self,
        *,
        skip_dirs: frozenset[str] | set[str] | None = None,
        suffixes: frozenset[str] | set[str] | None = None,
        scan_roots: tuple[str, ...] | None = None,
        min_prefix_len: int = _MIN_PREFIX_LEN,
    ) -> None:
        self._skip_dirs = frozenset(skip_dirs) if skip_dirs is not None else _DEFAULT_SKIP_DIRS
        self._suffixes = frozenset(suffixes) if suffixes is not None else _DEFAULT_SUFFIXES
        self._scan_roots = scan_roots if scan_roots is not None else _DEFAULT_SCAN_ROOTS
        self._min_prefix_len = min_prefix_len
        self._project_counts: Counter[str] = Counter()
        self._buffer_counts: Counter[str] = Counter()
        self._root: Path | None = None

    def backend_id(self) -> str:
        return "prefix"

    def set_buffer_text(self, text: str) -> None:
        self._buffer_counts = Counter(tokenize_identifiers(text))

    def index_project(self, root: Path) -> int:
        root = root.resolve()
        self._root = root
        counts: Counter[str] = Counter()
        for rel in self._scan_roots:
            base = root / rel
            if not base.is_dir():
                continue
            self._walk(base, counts)
        self._project_counts = counts
        return sum(counts.values())

    def reindex_file(self, path: Path, text: str | None = None) -> None:
        resolved = path.resolve()
        if not self._should_index_path(resolved):
            return
        if text is None:
            try:
                text = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return
        tokens = tokenize_identifiers(text)
        if not tokens:
            return
        self._project_counts.update(tokens)

    def _should_index_path(self, path: Path) -> bool:
        if path.suffix.lower() not in self._suffixes:
            return False
        root = self._root
        if root is None:
            return True
        try:
            rel = path.relative_to(root)
        except ValueError:
            return False
        parts = rel.parts
        if not parts:
            return False
        if parts[0] not in self._scan_roots:
            return False
        return not any(part in self._skip_dirs for part in parts[:-1])

    def _walk(self, base: Path, counts: Counter[str]) -> None:
        try:
            entries = list(base.iterdir())
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if name in self._skip_dirs or name.startswith("."):
                continue
            if entry.is_dir():
                self._walk(entry, counts)
                continue
            if entry.suffix.lower() not in self._suffixes:
                continue
            try:
                text = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            counts.update(tokenize_identifiers(text))

    def complete(self, query: CompletionQuery) -> CompletionCandidate | None:
        prefix = query.prefix
        if len(prefix) < self._min_prefix_len:
            return None
        if query.suffix and (query.suffix[0].isalnum() or query.suffix[0] == "_"):
            return None
        leaf = prefix.rsplit(".", 1)[-1]
        if len(leaf) < self._min_prefix_len:
            return None
        best = _best_token(leaf, self._buffer_counts)
        source = "buffer"
        if best is None:
            best = _best_token(leaf, self._project_counts)
            source = "project"
        if best is None:
            return None
        remainder = best[len(leaf) :]
        if not remainder:
            return None
        return CompletionCandidate(text=remainder, source=source)
