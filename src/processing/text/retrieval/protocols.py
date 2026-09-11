from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .types import RetrievalQuery, RetrievalResult


@runtime_checkable
class RagRetriever(Protocol):
    """Query-time backend. Chunk shape is backend-defined and callers use extras for arbitrary payloads."""

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult: ...

    def backend_id(self) -> str: ...


@runtime_checkable
class RagIndexer(Protocol):
    """Ingest/build side. Implementations vary by provider (LEANN vs FAISS vs hosted APIs)."""

    def backend_id(self) -> str: ...

    def add_documents(self, documents: Mapping[str, Any] | list[Mapping[str, Any]]) -> None:
        """Append or ingest documents depending on indexer semantics."""
        ...
