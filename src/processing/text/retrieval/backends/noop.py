from __future__ import annotations

from ..protocols import RagRetriever
from ..types import RetrievalQuery, RetrievalResult, empty_result


class NoopRetriever(RagRetriever):
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        _ = query
        return empty_result()

    def backend_id(self) -> str:
        return "noop"
