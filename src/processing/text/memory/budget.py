from __future__ import annotations

import os
from dataclasses import dataclass, field

from .tiers import MemoryTier

DEFAULT_TOTAL_CHARS = 4000
DEFAULT_RECALL_TURNS = 6

_TIER_FRACTION: dict[MemoryTier, float] = {
    "semantic": 0.35,
    "procedural": 0.15,
    "working": 0.20,
    "episodic": 0.10,
    "retrieved": 0.20,
}


@dataclass
class MemoryBudget:
    total_chars: int = DEFAULT_TOTAL_CHARS
    recall_turns: int = DEFAULT_RECALL_TURNS

    def cap_for(self, tier: MemoryTier) -> int:
        fraction = _TIER_FRACTION.get(tier, 0.2)
        return max(200, int(self.total_chars * fraction))


def budget_from_env(recall_turns: int | None = None) -> MemoryBudget:
    raw_total = os.environ.get("SOPHON_MEMORY_BUDGET_CHARS", "").strip()
    try:
        total = int(raw_total) if raw_total else DEFAULT_TOTAL_CHARS
    except ValueError:
        total = DEFAULT_TOTAL_CHARS
    if recall_turns is None:
        raw_recall = os.environ.get("SOPHON_MEMORY_RECALL_TURNS", "").strip()
        try:
            recall = int(raw_recall) if raw_recall else DEFAULT_RECALL_TURNS
        except ValueError:
            recall = DEFAULT_RECALL_TURNS
    else:
        recall = recall_turns
    return MemoryBudget(total_chars=max(500, total), recall_turns=max(0, recall))


@dataclass
class MemoryPackSection:
    tier: MemoryTier
    header: str
    lines: list[str] = field(default_factory=list)
    dropped: int = 0


@dataclass
class MemoryPack:
    sections: list[MemoryPackSection] = field(default_factory=list)
    total_dropped: int = 0

    def is_empty(self) -> bool:
        return not any(section.lines for section in self.sections)


def fit_lines(lines: list[str], cap_chars: int) -> tuple[list[str], int]:
    kept: list[str] = []
    used = 0
    dropped = 0
    for line in lines:
        cost = len(line) + 1
        if used + cost > cap_chars and kept:
            dropped += 1
            continue
        kept.append(line)
        used += cost
    return kept, dropped
