from __future__ import annotations

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://www.wikidata.org/w/api.php"


class WikidataSource(SearchSource):
    name = "wikidata"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "action": "wbsearchentities",
                    "search": query,
                    "language": "en",
                    "uselang": "en",
                    "type": "item",
                    "limit": limit,
                    "format": "json",
                },
            ),
            source=self.name,
        )
        rows = payload.get("search") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        hits: list[SearchHit] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            entity_id = str(row.get("id") or "").strip()
            url = str(row.get("concepturi") or row.get("url") or "").strip()
            if url and url.startswith("//"):
                url = "https:" + url
            hits.append(
                SearchHit(
                    title=str(row.get("label") or entity_id or "(untitled)").strip(),
                    url=url,
                    snippet=str(row.get("description") or "").strip(),
                    extras={"id": entity_id} if entity_id else {},
                )
            )
        return [hit for hit in hits if hit.url]


SOURCE = WikidataSource()
