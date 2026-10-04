from __future__ import annotations

from integrations.search.http import fetch_json, search_contact, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://api.openalex.org/works"


class OpenAlexSource(SearchSource):
    name = "openalex"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "search": query,
                    "per-page": limit,
                    "mailto": search_contact() or None,
                },
            ),
            source=self.name,
        )
        rows = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        return parse_openalex_works(rows, limit=limit)


def parse_openalex_works(rows: list[object], *, limit: int) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        openalex_id = str(row.get("id") or "").strip()
        doi = str(row.get("doi") or "").strip()
        url = doi or openalex_id
        extras: dict[str, str] = {}
        if doi:
            extras["doi"] = doi
        year = row.get("publication_year")
        if year is not None:
            extras["year"] = str(year)
        authors = _authorships(row.get("authorships"))
        if authors:
            extras["authors"] = authors
        if openalex_id:
            extras["id"] = openalex_id.rsplit("/", 1)[-1]
        snippet = _abstract(row.get("abstract_inverted_index"))
        if url:
            hits.append(
                SearchHit(
                    title=str(row.get("title") or row.get("display_name") or "").strip()
                    or "(untitled)",
                    url=url,
                    snippet=snippet,
                    extras=extras,
                )
            )
    return hits


def _authorships(rows: object) -> str:
    if not isinstance(rows, list):
        return ""
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        author = row.get("author")
        if isinstance(author, dict):
            name = str(author.get("display_name") or "").strip()
            if name:
                names.append(name)
    return ", ".join(names)


def _abstract(index: object) -> str:
    if not isinstance(index, dict):
        return ""
    placed: list[tuple[int, str]] = []
    for word, locs in index.items():
        if not isinstance(locs, list):
            continue
        for loc in locs:
            try:
                placed.append((int(loc), str(word)))
            except (TypeError, ValueError):
                continue
    placed.sort()
    return " ".join(word for _, word in placed)[:800]


SOURCE = OpenAlexSource()
