from __future__ import annotations

from abc import ABC, abstractmethod

from integrations.search.types import SearchHit


class SearchSource(ABC):
    name: str
    min_interval_s: float = 0.0

    def available(self) -> bool:
        return True

    @abstractmethod
    def search(self, query: str, limit: int) -> list[SearchHit]:
        raise NotImplementedError
