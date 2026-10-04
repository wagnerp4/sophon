from __future__ import annotations

from integrations.search.http import fetch_json, search_contact, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class PubmedSource(SearchSource):
    name = "pubmed"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        contact = search_contact()
        found = fetch_json(
            url_with_query(
                _ESEARCH,
                {
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": limit,
                    "tool": "sophon",
                    "email": contact or None,
                },
            ),
            source=self.name,
        )
        result = found.get("esearchresult") if isinstance(found, dict) else None
        ids = result.get("idlist") if isinstance(result, dict) else None
        if not isinstance(ids, list) or not ids:
            return []
        id_csv = ",".join(str(item) for item in ids[:limit])
        summary = fetch_json(
            url_with_query(
                _ESUMMARY,
                {
                    "db": "pubmed",
                    "id": id_csv,
                    "retmode": "json",
                    "tool": "sophon",
                    "email": contact or None,
                },
            ),
            source=self.name,
        )
        return _hits_from_summary(summary, ids[:limit])


def _hits_from_summary(payload: object, ids: list[object]) -> list[SearchHit]:
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        return []
    hits: list[SearchHit] = []
    for pmid in ids:
        key = str(pmid)
        row = result.get(key)
        if not isinstance(row, dict):
            continue
        extras: dict[str, str] = {"id": key}
        year = str(row.get("pubdate") or "")[:4]
        if year.isdigit():
            extras["year"] = year
        authors = _pubmed_authors(row.get("authors"))
        if authors:
            extras["authors"] = authors
        doi = ""
        article_ids = row.get("articleids")
        if isinstance(article_ids, list):
            for item in article_ids:
                if isinstance(item, dict) and str(item.get("idtype") or "") == "doi":
                    doi = str(item.get("value") or "").strip()
                    break
        if doi:
            extras["doi"] = doi
        hits.append(
            SearchHit(
                title=str(row.get("title") or "").strip() or "(untitled)",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{key}/",
                snippet=str(row.get("source") or "").strip(),
                extras=extras,
            )
        )
    return hits


def _pubmed_authors(authors: object) -> str:
    if not isinstance(authors, list):
        return ""
    names: list[str] = []
    for author in authors:
        if isinstance(author, dict):
            name = str(author.get("name") or "").strip()
            if name:
                names.append(name)
    return ", ".join(names)


SOURCE = PubmedSource()
