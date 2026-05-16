from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def empty_result() -> RetrievalResult:
    return RetrievalResult()
