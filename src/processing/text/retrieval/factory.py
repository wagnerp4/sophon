from __future__ import annotations

from typing import Literal

from .backends.leann_bridge import load_retriever_from_entrypoint, resolve_leann_retriever
from .backends.noop import NoopRetriever
from .protocols import RagRetriever

RagRetrieverId = Literal["noop", "leann"]


def rag_retriever_ids() -> tuple[str, ...]:
    return ("noop", "leann")


def load_rag_retriever(
    backend_id: RagRetrieverId | str = "noop",
    *,
    entrypoint: str | None = None,
    native_index_path: str | None = None,
) -> RagRetriever:
    """Select a retrieval backend. Use entrypoint= for dotted module paths and native_index_path= or MITHRIL_LEANN_INDEX for LEANN."""
    if entrypoint:
        return load_retriever_from_entrypoint(entrypoint.strip())
    if backend_id == "noop":
        return NoopRetriever()
    if backend_id == "leann":
        return resolve_leann_retriever(native_index_path=native_index_path)
    raise KeyError(f"unknown retrieval backend {backend_id!r}")
