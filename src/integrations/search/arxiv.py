from __future__ import annotations

import xml.etree.ElementTree as ET

from integrations.search.http import fetch_text, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ATOM = "http://www.w3.org/2005/Atom"
_ENDPOINT = "https://export.arxiv.org/api/query"


class ArxivSource(SearchSource):
    name = "arxiv"
    min_interval_s = 3.0

    def search(self, query: str, limit: int) -> list[SearchHit]:
        xml_text = fetch_text(
            url_with_query(
                _ENDPOINT,
                {
                    "search_query": arxiv_search_query(query),
                    "start": 0,
                    "max_results": limit,
                },
            ),
            source=self.name,
            min_interval_s=self.min_interval_s,
            headers={"Accept": "application/atom+xml, application/xml, text/xml, */*"},
        )
        return parse_arxiv_atom(xml_text, limit=limit)


def arxiv_search_query(query: str) -> str:
    q = query.strip()
    if not q:
        return "all:"
    if q.startswith("all:") or q.startswith("ti:") or q.startswith("abs:"):
        return q
    if " " in q and not (q.startswith('"') and q.endswith('"')):
        return f'all:"{q}"'
    return f"all:{q}"


def parse_arxiv_atom(xml_text: str, *, limit: int) -> list[SearchHit]:
    root = ET.fromstring(xml_text)
    hits: list[SearchHit] = []
    for entry in root.findall(f"{{{_ATOM}}}entry"):
        if len(hits) >= limit:
            break
        title = _text(entry.find(f"{{{_ATOM}}}title"))
        url = _text(entry.find(f"{{{_ATOM}}}id"))
        snippet = _text(entry.find(f"{{{_ATOM}}}summary"))
        published = _text(entry.find(f"{{{_ATOM}}}published"))
        authors = [
            _text(author.find(f"{{{_ATOM}}}name"))
            for author in entry.findall(f"{{{_ATOM}}}author")
        ]
        extras: dict[str, str] = {}
        year = published[:4] if published else ""
        if year:
            extras["year"] = year
        names = ", ".join(name for name in authors if name)
        if names:
            extras["authors"] = names
        arxiv_id = url.rsplit("/", 1)[-1] if url else ""
        if arxiv_id:
            extras["id"] = arxiv_id
        if url:
            hits.append(
                SearchHit(
                    title=title or "(untitled)",
                    url=url,
                    snippet=" ".join(snippet.split()),
                    extras=extras,
                )
            )
    return hits


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split()).strip()


SOURCE = ArxivSource()
