from __future__ import annotations

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarSource(SearchSource):
    name = "semanticscholar"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "query": query,
                    "limit": limit,
                    "fields": "title,url,abstract,year,authors,externalIds",
                },
            ),
            source=self.name,
        )
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        hits: list[SearchHit] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            paper_id = str(row.get("paperId") or "").strip()
            url = str(row.get("url") or "").strip()
            if not url and paper_id:
                url = f"https://www.semanticscholar.org/paper/{paper_id}"
            extras: dict[str, str] = {}
            year = row.get("year")
            if year is not None:
                extras["year"] = str(year)
            authors = _s2_authors(row.get("authors"))
            if authors:
                extras["authors"] = authors
            external = row.get("externalIds")
            if isinstance(external, dict):
                doi = str(external.get("DOI") or "").strip()
                if doi:
                    extras["doi"] = doi
            if paper_id:
                extras["id"] = paper_id
            snippet = str(row.get("abstract") or "").strip()
            snippet = " ".join(snippet.split())[:800]
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


def _s2_authors(authors: object) -> str:
    if not isinstance(authors, list):
        return ""
    names: list[str] = []
    for author in authors:
        if isinstance(author, dict):
            name = str(author.get("name") or "").strip()
            if name:
                names.append(name)
    return ", ".join(names)


SOURCE = SemanticScholarSource()
