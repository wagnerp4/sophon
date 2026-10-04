from __future__ import annotations

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://api.duckduckgo.com/"


class DdgSource(SearchSource):
    name = "ddg"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "q": query,
                    "format": "json",
                    "no_redirect": 1,
                    "no_html": 1,
                    "skip_disambig": 1,
                },
            ),
            source=self.name,
        )
        if not isinstance(payload, dict):
            return []
        hits: list[SearchHit] = []
        abstract = str(payload.get("Abstract") or payload.get("AbstractText") or "").strip()
        abstract_url = str(payload.get("AbstractURL") or "").strip()
        heading = str(payload.get("Heading") or query).strip()
        if abstract and abstract_url:
            hits.append(
                SearchHit(
                    title=heading or "(untitled)",
                    url=abstract_url,
                    snippet=abstract,
                )
            )
        related = payload.get("RelatedTopics") or []
        if isinstance(related, list):
            _collect_related(related, hits, limit)
        return hits[:limit]


def _collect_related(rows: list[object], hits: list[SearchHit], limit: int) -> None:
    for row in rows:
        if len(hits) >= limit:
            return
        if not isinstance(row, dict):
            continue
        nested = row.get("Topics")
        if isinstance(nested, list):
            _collect_related(nested, hits, limit)
            continue
        url = str(row.get("FirstURL") or "").strip()
        text = str(row.get("Text") or "").strip()
        if not url:
            continue
        hits.append(
            SearchHit(
                title=text.split(" - ", 1)[0].strip() or "(untitled)",
                url=url,
                snippet=text,
            )
        )


SOURCE = DdgSource()
