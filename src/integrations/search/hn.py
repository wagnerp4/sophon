from __future__ import annotations

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://hn.algolia.com/api/v1/search"


class HnSource(SearchSource):
    name = "hn"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "query": query,
                    "hitsPerPage": limit,
                },
            ),
            source=self.name,
        )
        rows = payload.get("hits") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        hits: list[SearchHit] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            object_id = str(row.get("objectID") or "").strip()
            url = str(row.get("url") or row.get("story_url") or "").strip()
            if not url and object_id:
                url = f"https://news.ycombinator.com/item?id={object_id}"
            extras: dict[str, str] = {}
            if object_id:
                extras["id"] = object_id
            author = str(row.get("author") or "").strip()
            if author:
                extras["authors"] = author
            snippet = str(
                row.get("story_text") or row.get("comment_text") or row.get("title") or ""
            ).strip()
            snippet = " ".join(snippet.split())[:800]
            title = str(row.get("title") or row.get("story_title") or "").strip()
            if url:
                hits.append(
                    SearchHit(
                        title=title or "(untitled)",
                        url=url,
                        snippet=snippet,
                        extras=extras,
                    )
                )
        return hits


SOURCE = HnSource()
