from __future__ import annotations

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_OPENSEARCH = "https://en.wikipedia.org/w/api.php"
_EXTRACT = "https://en.wikipedia.org/w/api.php"


class WikipediaSource(SearchSource):
    name = "wikipedia"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _OPENSEARCH,
                {
                    "action": "opensearch",
                    "search": query,
                    "limit": limit,
                    "namespace": 0,
                    "format": "json",
                },
            ),
            source=self.name,
        )
        hits = parse_wikipedia_opensearch(payload, limit=limit)
        if not hits:
            return []
        extract = _first_extract(hits[0].title)
        if extract:
            first = hits[0]
            hits[0] = SearchHit(
                title=first.title,
                url=first.url,
                snippet=extract,
                extras=first.extras,
            )
        return hits


def parse_wikipedia_opensearch(payload: object, *, limit: int) -> list[SearchHit]:
    titles: list[str] = []
    snippets: list[str] = []
    urls: list[str] = []
    if isinstance(payload, list) and len(payload) >= 4:
        titles = [str(item) for item in (payload[1] or [])]
        snippets = [str(item) for item in (payload[2] or [])]
        urls = [str(item) for item in (payload[3] or [])]
    hits: list[SearchHit] = []
    for index, title in enumerate(titles[:limit]):
        url = urls[index] if index < len(urls) else ""
        snippet = snippets[index] if index < len(snippets) else ""
        if not url.strip():
            continue
        hits.append(
            SearchHit(
                title=title.strip() or "(untitled)",
                url=url.strip(),
                snippet=snippet.strip(),
                extras={"license": "CC BY-SA"},
            )
        )
    return hits


def _first_extract(title: str) -> str:
    payload = fetch_json(
        url_with_query(
            _EXTRACT,
            {
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "format": "json",
                "titles": title,
            },
        ),
        source="wikipedia",
    )
    if not isinstance(payload, dict):
        return ""
    pages = (payload.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return ""
    for page in pages.values():
        if isinstance(page, dict):
            text = str(page.get("extract") or "").strip()
            if text:
                return text[:800]
    return ""


SOURCE = WikipediaSource()
