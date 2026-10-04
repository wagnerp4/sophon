from __future__ import annotations

from integrations.search.dispatch import hits_from_maps
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit


class SearxngSource(SearchSource):
    name = "searxng"

    def available(self) -> bool:
        from integrations.google.searxng import searxng_configured

        return searxng_configured()

    def search(self, query: str, limit: int) -> list[SearchHit]:
        from integrations.google.searxng import fetch_searxng_hits

        if not self.available():
            raise RuntimeError(
                "SearXNG is not configured. Set SOPHON_SEARXNG_URL to a running instance."
            )
        return hits_from_maps(fetch_searxng_hits(query, limit=limit))


SOURCE = SearxngSource()
