from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RetrievalDecision(str, Enum):
    SKIP = "skip"
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


_SKIP_EXACT = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "ok",
        "okay",
        "k",
        "kk",
        "yes",
        "yep",
        "yeah",
        "no",
        "nope",
        "bye",
        "goodbye",
        "good night",
        "good morning",
        "gm",
        "cool",
        "nice",
        "great",
        "sure",
        "alright",
        "got it",
        "understood",
    }
)

_SKIP_PREFIXES = (
    "thanks ",
    "thank you ",
    "hi ",
    "hello ",
    "hey ",
)

_MULTI_HOP_PATTERNS = (
    re.compile(r"\b(compare|comparison|versus|vs\.?)\b", re.I),
    re.compile(r"\b(across|throughout|overall|themes?|summarize|summary)\b", re.I),
    re.compile(r"\b(relate|related|relationship|connection|between)\b", re.I),
    re.compile(r"\b(how (do|does|did|are|is).{0,40}\b(affect|influence|differ|connect)\b)", re.I),
    re.compile(r"\b(multi[- ]?hop|multi[- ]?document|cross[- ]?document)\b", re.I),
    re.compile(r"\b(and|vs|versus)\b.+\b(and|vs|versus)\b", re.I),
)


@dataclass(frozen=True)
class AdaptiveDecision:
    decision: RetrievalDecision
    reason: str


def normalize_query(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def decide_retrieval(query: str, *, structure_available: bool = False) -> AdaptiveDecision:
    """Heuristic Adaptive-RAG gate: skip / single-hop / multi-hop.

    Until a trained classifier exists this is deterministic text policy only.
    multi_hop is only returned when a structure retriever is available; otherwise
    multi-hop cues fall through to single_hop so LEANN still runs.
    """
    norm = normalize_query(query)
    if not norm:
        return AdaptiveDecision(RetrievalDecision.SKIP, "empty_query")
    if norm in _SKIP_EXACT:
        return AdaptiveDecision(RetrievalDecision.SKIP, "phatic_exact")
    if len(norm) <= 3 and not any(ch.isalnum() for ch in norm):
        return AdaptiveDecision(RetrievalDecision.SKIP, "too_short")
    for prefix in _SKIP_PREFIXES:
        if norm.startswith(prefix) and len(norm) < 24:
            return AdaptiveDecision(RetrievalDecision.SKIP, "phatic_prefix")
    for pattern in _MULTI_HOP_PATTERNS:
        if pattern.search(norm):
            if structure_available:
                return AdaptiveDecision(RetrievalDecision.MULTI_HOP, f"pattern:{pattern.pattern}")
            return AdaptiveDecision(RetrievalDecision.SINGLE_HOP, "multi_hop_cue_no_structure")
    return AdaptiveDecision(RetrievalDecision.SINGLE_HOP, "default")
