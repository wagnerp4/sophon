from __future__ import annotations

import re

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://zenodo.org/api/records"
_TAG = re.compile(r"<[^>]+>")


class ZenodoSource(SearchSource):
    name = "zenodo"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "q": query,
                    "size": limit,
                    "sort": "mostrecent",
                },
            ),
            source=self.name,
        )
        return parse_zenodo_records(payload, limit=limit)


def parse_zenodo_records(payload: object, *, limit: int) -> list[SearchHit]:
    listing = payload.get("hits") if isinstance(payload, dict) else None
    rows = listing.get("hits") if isinstance(listing, dict) else None
    if not isinstance(rows, list):
        return []
    hits: list[SearchHit] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        links = row.get("links") if isinstance(row.get("links"), dict) else {}
        doi = str(row.get("doi") or meta.get("doi") or "").strip()
        url = str(links.get("html") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"
        extras: dict[str, str] = {}
        if doi:
            extras["doi"] = doi
        published = str(meta.get("publication_date") or "").strip()
        if published[:4].isdigit():
            extras["year"] = published[:4]
        creators = meta.get("creators")
        if isinstance(creators, list):
            names = [
                str(item.get("name") or "").strip()
                for item in creators
                if isinstance(item, dict)
            ]
            authors = ", ".join(name for name in names if name)
            if authors:
                extras["authors"] = authors
        snippet = _plain(str(meta.get("description") or ""))
        if url:
            hits.append(
                SearchHit(
                    title=str(meta.get("title") or "").strip() or "(untitled)",
                    url=url,
                    snippet=snippet[:800],
                    extras=extras,
                )
            )
    return hits


def _plain(text: str) -> str:
    return " ".join(_TAG.sub(" ", text).split())


SOURCE = ZenodoSource()
