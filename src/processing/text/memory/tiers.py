from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

MemoryTier = Literal["working", "episodic", "semantic", "procedural", "retrieved"]
Confidence = Literal["observed", "decided", "hypothesis"]

MEMORY_TIERS: tuple[MemoryTier, ...] = (
    "working",
    "episodic",
    "semantic",
    "procedural",
    "retrieved",
)
CONFIDENCES: tuple[Confidence, ...] = ("observed", "decided", "hypothesis")
PROMOTABLE_CONFIDENCES: tuple[Confidence, ...] = ("observed", "decided")

_TIER_PREFIX: dict[str, MemoryTier] = {
    "w": "working",
    "e": "episodic",
    "s": "semantic",
    "p": "procedural",
    "r": "retrieved",
}


def normalize_tier(raw: str | None, default: MemoryTier = "working") -> MemoryTier:
    value = str(raw or "").strip().lower()
    if value in MEMORY_TIERS:
        return value  # type: ignore[return-value]
    aliases = {
        "scratch": "working",
        "session": "working",
        "note": "working",
        "notes": "working",
        "fact": "semantic",
        "facts": "semantic",
        "durable": "semantic",
        "procedure": "procedural",
        "procedures": "procedural",
        "skill": "procedural",
        "episode": "episodic",
        "episodes": "episodic",
        "history": "episodic",
    }
    return aliases.get(value, default)


def normalize_confidence(raw: str | None, default: Confidence = "hypothesis") -> Confidence:
    value = str(raw or "").strip().lower()
    if value in CONFIDENCES:
        return value  # type: ignore[return-value]
    aliases = {
        "verified": "observed",
        "seen": "observed",
        "evidence": "observed",
        "chosen": "decided",
        "decision": "decided",
        "guess": "hypothesis",
        "maybe": "hypothesis",
    }
    return aliases.get(value, default)


def entry_id(tier: MemoryTier, text: str) -> str:
    digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:8]
    return f"{tier[0]}{digest}"


@dataclass
class MemoryEntry:
    id: str
    tier: MemoryTier
    text: str
    confidence: Confidence = "hypothesis"
    source: str = "system"
    created_at: float = 0.0
    revoked_at: float | None = None
    tags: list[str] = field(default_factory=list)
    user_id: str | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def with_id(self) -> "MemoryEntry":
        if not self.id:
            self.id = entry_id(self.tier, self.text)
        return self
