from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .budget import MemoryBudget, MemoryPack, MemoryPackSection, fit_lines
from .markdown_store import MarkdownMemoryStore
from .sqlite_memory import SqliteMemoryStore
from .tiers import (
    Confidence,
    MemoryEntry,
    MemoryTier,
    PROMOTABLE_CONFIDENCES,
    entry_id,
    normalize_tier,
)
from .types import MemoryScope


@dataclass
class PromotionProposal:
    path: Path
    before: str
    after: str
    promoted_ids: list[str]
    skipped_ids: list[str]


class MemoryLayer:
    def __init__(
        self,
        sqlite: SqliteMemoryStore,
        markdown: MarkdownMemoryStore,
        scope: MemoryScope,
    ) -> None:
        self.sqlite = sqlite
        self.markdown = markdown
        self.scope = scope
        self.markdown.ensure_scaffold()

    def write_scratch(
        self,
        text: str,
        confidence: Confidence = "hypothesis",
        source: str = "model",
        tags: list[str] | None = None,
    ) -> tuple[MemoryEntry | None, str]:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None, "empty note ignored"
        if self.markdown.contains_secret(cleaned):
            redacted, _ = self.markdown.redact(cleaned)
            cleaned = redacted
            note = " (secret redacted)"
        else:
            note = ""
        entry = MemoryEntry(
            id=entry_id("working", cleaned),
            tier="working",
            text=cleaned,
            confidence=confidence,
            source=source,
            created_at=time.time(),
            tags=[t.strip() for t in (tags or []) if str(t).strip()],
            user_id=self.scope.user_id,
        )
        stored = self.markdown.append_scratch(self.scope.session_id, entry)
        return stored, f"noted {stored.id}{note}"

    def list(self, tier: MemoryTier) -> list[MemoryEntry]:
        if tier == "semantic":
            return [e for e in self.markdown.read_facts() if e.active]
        if tier == "procedural":
            return [e for e in self.markdown.read_procedures() if e.active]
        if tier == "working":
            return self.markdown.read_scratch(self.scope.session_id)
        if tier == "episodic":
            rows = self.sqlite.load_recent_episodes(self.scope, 20)
            return [
                MemoryEntry(
                    id=f"e{row['rowid']}",
                    tier="episodic",
                    text=row["summary"],
                    confidence="observed",
                    source="system",
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        return []

    def search(self, query: str, tiers: list[MemoryTier] | None = None, limit: int = 10) -> list[MemoryEntry]:
        needle = str(query or "").strip().lower()
        wanted = tiers or ["semantic", "procedural", "working", "episodic"]
        hits: list[MemoryEntry] = []
        for tier in wanted:
            for entry in self.list(tier):
                if not needle or needle in entry.text.lower():
                    hits.append(entry)
        # TODO: FTS5 / LEANN ranking. Substring order is tier-then-recency for now.
        return hits[: max(1, limit)]

    def get(self, tier: MemoryTier, entry_id_value: str) -> MemoryEntry | None:
        for entry in self.list(tier):
            if entry.id == entry_id_value:
                return entry
        return None

    def note(self, text: str, confidence: Confidence = "decided") -> tuple[MemoryEntry | None, str]:
        return self.write_scratch(text, confidence=confidence, source="human")

    def propose_promotion(self, ids: list[str]) -> PromotionProposal:
        wanted = {i.strip() for i in ids if i.strip()}
        scratch = self.markdown.read_scratch(self.scope.session_id)
        facts = self.markdown.read_facts()
        before = self.markdown.render_facts(facts)
        promoted: list[str] = []
        skipped: list[str] = []
        for entry in scratch:
            if entry.id not in wanted:
                continue
            if entry.confidence not in PROMOTABLE_CONFIDENCES:
                skipped.append(f"{entry.id} (hypothesis)")
                continue
            if self.markdown.contains_secret(entry.text):
                skipped.append(f"{entry.id} (secret)")
                continue
            if any(f.text.strip() == entry.text.strip() for f in facts):
                skipped.append(f"{entry.id} (duplicate)")
                continue
            fact = MemoryEntry(
                id=entry_id("semantic", entry.text),
                tier="semantic",
                text=entry.text,
                confidence=entry.confidence,
                source="human",
                created_at=time.time(),
                tags=entry.tags,
                user_id=self.scope.user_id,
            )
            facts.append(fact)
            promoted.append(entry.id)
        after = self.markdown.render_facts(facts)
        return PromotionProposal(
            path=self.markdown.facts_path,
            before=before,
            after=after,
            promoted_ids=promoted,
            skipped_ids=skipped,
        )

    def sync_facts_index(self) -> None:
        for entry in self.markdown.read_facts():
            entry.user_id = entry.user_id or self.scope.user_id
            try:
                self.sqlite.upsert_fact(entry)
            except Exception:
                continue

    def revoke(self, fact_id: str) -> tuple[bool, str]:
        target = fact_id.strip()
        facts = self.markdown.read_facts()
        kept = [f for f in facts if f.id != target]
        if len(kept) == len(facts):
            return False, f"no durable fact with id {target!r}"
        self.markdown.write_facts(kept)
        try:
            self.sqlite.revoke_fact(target)
        except Exception:
            pass
        return True, f"revoked {target}"

    def compact(self) -> str:
        scratch = self.markdown.read_scratch(self.scope.session_id)
        if not scratch:
            return "scratch already empty"
        summary = "; ".join(e.text.strip() for e in scratch)
        summary = summary[:2000]
        self.sqlite.add_episode(self.scope, summary, len(scratch))
        removed = self.markdown.clear_scratch(self.scope.session_id)
        return f"compacted {removed} scratch note(s) into one episode"

    def rotate_session(self, new_session_id: str) -> MemoryScope:
        old_id = self.scope.session_id
        notes = self.markdown.read_scratch(old_id)
        if notes:
            summary = "; ".join(e.text.strip() for e in notes)[:2000]
            try:
                self.sqlite.add_episode(self.scope, summary, len(notes))
            except Exception:
                pass
            self.markdown.clear_scratch(old_id)
        self.scope = MemoryScope(session_id=new_session_id, user_id=self.scope.user_id)
        if notes:
            self.markdown.write_scratch(new_session_id, notes)
        return self.scope

    def record_episode_on_close(self, transcript_summary: str, turn_count: int) -> None:
        try:
            self.sqlite.add_episode(self.scope, transcript_summary, turn_count)
        except Exception:
            pass

    def status(self) -> dict[str, int]:
        return {
            "semantic": len(self.list("semantic")),
            "procedural": len(self.list("procedural")),
            "working": len(self.list("working")),
            "episodic": len(self.sqlite.load_recent_episodes(self.scope, 1000)),
        }

    def pack(self, query: str, budget: MemoryBudget) -> MemoryPack:
        sections: list[MemoryPackSection] = []
        total_dropped = 0

        def add(tier: MemoryTier, header: str, raw_lines: list[str]) -> None:
            nonlocal total_dropped
            if not raw_lines:
                return
            kept, dropped = fit_lines(raw_lines, budget.cap_for(tier))
            total_dropped += dropped
            sections.append(MemoryPackSection(tier=tier, header=header, lines=kept, dropped=dropped))

        facts = [f"[{e.confidence}] {e.text}" for e in self.list("semantic")]
        add("semantic", "Durable facts", facts)

        procs = [e.text for e in self.list("procedural")]
        add("procedural", "Procedures", procs)

        scratch = [f"[{e.confidence}] {e.text}" for e in self.list("working")]
        add("working", "Session notes", scratch)

        episodes = self.sqlite.load_recent_episodes(self.scope, 5)
        episode_lines = [row["summary"] for row in episodes]
        add("episodic", "Earlier sessions", episode_lines)

        if budget.recall_turns > 0:
            turns = self.sqlite.load_recent_turns(self.scope, budget.recall_turns)
            turn_lines = [f"{t.role}: {t.content.strip()}" for t in turns]
            add("retrieved", "Earlier turns", turn_lines)

        return MemoryPack(sections=sections, total_dropped=total_dropped)
