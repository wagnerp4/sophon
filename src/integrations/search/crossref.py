from __future__ import annotations

from integrations.search.http import fetch_json, search_contact, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://api.crossref.org/works"


class CrossrefSource(SearchSource):
    name = "crossref"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "query": query,
                    "rows": limit,
                    "mailto": search_contact() or None,
                },
            ),
            source=self.name,
        )
        message = payload.get("message") if isinstance(payload, dict) else None
        items = message.get("items") if isinstance(message, dict) else None
        if not isinstance(items, list):
            return []
        hits: list[SearchHit] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            titles = item.get("title") or []
            title = titles[0] if isinstance(titles, list) and titles else item.get("title")
            doi = str(item.get("DOI") or "").strip()
            url = str(item.get("URL") or "").strip()
            if not url and doi:
                url = f"https://doi.org/{doi}"
            extras: dict[str, str] = {}
            if doi:
                extras["doi"] = doi
            year = _issued_year(item.get("issued"))
            if year:
                extras["year"] = year
            authors = _author_names(item.get("author"))
            if authors:
                extras["authors"] = authors
            snippet = ""
            abstract = item.get("abstract")
            if isinstance(abstract, str):
                snippet = " ".join(abstract.split())[:800]
            if url:
                hits.append(
                    SearchHit(
                        title=str(title or "").strip() or "(untitled)",
                        url=url,
                        snippet=snippet,
                        extras=extras,
                    )
                )
        return hits


def _issued_year(issued: object) -> str:
    if not isinstance(issued, dict):
        return ""
    parts = issued.get("date-parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
        return str(parts[0][0])
    return ""


def _author_names(authors: object) -> str:
    if not isinstance(authors, list):
        return ""
    names: list[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        family = str(author.get("family") or "").strip()
        given = str(author.get("given") or "").strip()
        label = " ".join(part for part in (given, family) if part)
        if label:
            names.append(label)
    return ", ".join(names)


SOURCE = CrossrefSource()
