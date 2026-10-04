from __future__ import annotations

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://developer.mozilla.org/api/v1/search"


class MdnSource(SearchSource):
    name = "mdn"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "q": query,
                    "locale": "en-US",
                },
            ),
            source=self.name,
        )
        rows = payload.get("documents") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        hits: list[SearchHit] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            path = str(row.get("mdn_url") or "").strip()
            url = path
            if path.startswith("/"):
                url = f"https://developer.mozilla.org{path}"
            extras: dict[str, str] = {}
            slug = str(row.get("slug") or "").strip()
            if slug:
                extras["id"] = slug
            if url:
                hits.append(
                    SearchHit(
                        title=str(row.get("title") or "").strip() or "(untitled)",
                        url=url,
                        snippet=str(row.get("summary") or "").strip(),
                        extras=extras,
                    )
                )
        return hits


SOURCE = MdnSource()
