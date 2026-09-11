from __future__ import annotations

from typing import Iterable

from ..retrieval.types import RetrievalResult

from .types import ContextBuildResult, Message

_DEFAULT_RETRIEVAL_HEADER = "Retrieved context (use only if relevant):"
_DEFAULT_MEMORY_HEADER = "Earlier conversation summary:"


def _chunk_to_text(chunk: dict, max_chars: int) -> str:
    text = chunk.get("text") or chunk.get("content") or chunk.get("value") or ""
    text_s = str(text).strip()
    if max_chars > 0 and len(text_s) > max_chars:
        text_s = text_s[: max_chars - 1].rstrip() + "\u2026"
    src = chunk.get("id") or chunk.get("source") or chunk.get("uri") or ""
    if src:
        return f"[{src}] {text_s}"
    return text_s


def render_retrieval_block(
    result: RetrievalResult | None,
    *,
    max_chunks: int = 5,
    max_chars_per_chunk: int = 800,
    header: str = _DEFAULT_RETRIEVAL_HEADER,
) -> str | None:
    """Format retrieval hits into a single block. Returns None when there is nothing useful to inject."""
    if result is None:
        return None
    chunks = list(result.chunks) if result.chunks else []
    if not chunks:
        return None
    selected = chunks[: max_chunks if max_chunks > 0 else len(chunks)]
    lines = [header]
    for i, chunk in enumerate(selected, start=1):
        lines.append(f"{i}. {_chunk_to_text(chunk, max_chars_per_chunk)}")
    return "\n".join(lines).strip()


def render_memory_block(
    turns: Iterable[dict | object],
    *,
    max_turns: int = 6,
    header: str = _DEFAULT_MEMORY_HEADER,
) -> str | None:
    """Render the trailing N turns (oldest first) as a compact text block."""
    materialized: list[tuple[str, str]] = []
    for turn in turns:
        role = getattr(turn, "role", None)
        content = getattr(turn, "content", None)
        if role is None and isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
        if not role or content is None:
            continue
        materialized.append((str(role), str(content)))
    if not materialized:
        return None
    selected = materialized[-max_turns:] if max_turns > 0 else materialized
    lines = [header]
    for role, content in selected:
        lines.append(f"- {role}: {content.strip()}")
    return "\n".join(lines).strip()


def build_messages_for_model(
    base_messages: list[Message],
    *,
    retrieval: RetrievalResult | None = None,
    memory_turns: Iterable[object] | None = None,
    system_text: str | None = None,
    max_retrieval_chunks: int = 5,
    max_retrieval_chars: int = 800,
    max_memory_turns: int = 6,
) -> ContextBuildResult:
    """
    Compose the message list passed to generate_response for this turn only.

    The returned list is a fresh copy of base_messages. Optional retrieval and memory blocks are
    injected as additional system messages right after any existing system prompt or system_text so
    the canonical chat transcript on the caller is not mutated.
    """
    out: list[Message] = []
    injected: list[str] = []
    if system_text is not None and str(system_text).strip():
        out.append({"role": "system", "content": str(system_text).strip()})

    retrieval_block = render_retrieval_block(
        retrieval,
        max_chunks=max_retrieval_chunks,
        max_chars_per_chunk=max_retrieval_chars,
    )
    memory_block = render_memory_block(memory_turns or (), max_turns=max_memory_turns)

    if memory_block:
        out.append({"role": "system", "content": memory_block})
        injected.append("memory")
    if retrieval_block:
        out.append({"role": "system", "content": retrieval_block})
        injected.append("retrieval")

    for msg in base_messages:
        if not isinstance(msg, dict):
            continue
        if system_text is not None and msg.get("role") == "system":
            continue
        out.append(dict(msg))

    return ContextBuildResult(
        messages=out,
        retrieval_used=bool(retrieval_block),
        memory_used=bool(memory_block),
        injected_blocks=injected,
    )
