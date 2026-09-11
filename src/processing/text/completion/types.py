from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompletionQuery:
    prefix: str
    suffix: str
    document_text: str
    path: Path | None = None
    language: str | None = None


@dataclass(frozen=True)
class CompletionCandidate:
    text: str
    source: str
