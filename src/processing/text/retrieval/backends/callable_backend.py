from __future__ import annotations

import copy
from collections.abc import Callable, Mapping

from ..protocols import RagRetriever
from ..types import RetrievalQuery, RetrievalResult, empty_result


class CallableRetriever(RagRetriever):
    """Wrap any user-facing callable(query: RetrievalQuery) -> RetrievalResult | mapping."""

    def __init__(
        self,
        impl: Callable[[RetrievalQuery], object],
        *,
        backend_id: str = "callable",
    ) -> None:
        self._impl = impl
        self._backend_id = backend_id

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        raw = self._impl(query)
        if isinstance(raw, RetrievalResult):
            return raw
        if isinstance(raw, Mapping):
            chunks = raw.get("chunks") or raw.get("documents") or raw.get("hits")
            if chunks is None and "texts" in raw:
                chunks = [{"text": t} for t in raw["texts"]] if isinstance(raw["texts"], list) else None
            chunk_list = list(chunks) if isinstance(chunks, list) else []
            typed_chunks: list[dict[str, object]] = []
            for item in chunk_list:
                if isinstance(item, Mapping):
                    typed_chunks.append(dict(item))
                else:
                    typed_chunks.append({"value": item})
            extras_payload = raw.get("extras") if isinstance(raw.get("extras"), Mapping) else {}
            merged = dict(extras_payload) if extras_payload else {}
            for key, value in raw.items():
                if key in ("chunks", "documents", "hits", "texts", "extras"):
                    continue
                merged[key] = copy.deepcopy(value)
            return RetrievalResult(chunks=typed_chunks, extras=merged)
        return empty_result()

    def backend_id(self) -> str:
        return self._backend_id
