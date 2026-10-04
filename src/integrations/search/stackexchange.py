from __future__ import annotations

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://api.stackexchange.com/2.3/search/advanced"


class StackexchangeSource(SearchSource):
    name = "stackexchange"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "site": "stackoverflow",
                    "q": query,
                    "pagesize": limit,
                    "order": "desc",
                    "sort": "relevance",
                    "filter": "default",
                },
            ),
            source=self.name,
        )
        rows = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        hits: list[SearchHit] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            url = str(row.get("link") or "").strip()
            extras: dict[str, str] = {}
            qid = row.get("question_id")
            if qid is not None:
                extras["id"] = str(qid)
            snippet = str(row.get("excerpt") or "").strip()
            if not snippet:
                tags = row.get("tags")
                if isinstance(tags, list):
                    snippet = ", ".join(str(tag) for tag in tags)
            if url:
                hits.append(
                    SearchHit(
                        title=str(row.get("title") or "").strip() or "(untitled)",
                        url=url,
                        snippet=snippet,
                        extras=extras,
                    )
                )
        return hits


SOURCE = StackexchangeSource()
