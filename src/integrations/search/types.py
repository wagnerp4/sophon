from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    extras: dict[str, str] = field(default_factory=dict)
