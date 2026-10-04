from __future__ import annotations

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePmcSource(SearchSource):
    name = "europepmc"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "query": query,
                    "format": "json",
                    "pageSize": limit,
                },
            ),
            source=self.name,
        )
        listing = payload.get("resultList") if isinstance(payload, dict) else None
        rows = listing.get("result") if isinstance(listing, dict) else None
        if not isinstance(rows, list):
            return []
        hits: list[SearchHit] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            pmid = str(row.get("pmid") or "").strip()
            pmcid = str(row.get("pmcid") or "").strip()
            doi = str(row.get("doi") or "").strip()
            url = ""
            if pmid:
                url = f"https://europepmc.org/article/MED/{pmid}"
            elif pmcid:
                url = f"https://europepmc.org/article/PMC/{pmcid}"
            elif doi:
                url = f"https://doi.org/{doi}"
            extras: dict[str, str] = {}
            if doi:
                extras["doi"] = doi
            year = str(row.get("pubYear") or "").strip()
            if year:
                extras["year"] = year
            authors = str(row.get("authorString") or "").strip()
            if authors:
                extras["authors"] = authors
            if pmid:
                extras["id"] = pmid
            snippet = str(row.get("abstractText") or "").strip()
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


SOURCE = EuropePmcSource()
