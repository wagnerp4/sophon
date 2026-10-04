from __future__ import annotations

import json
from typing import Any

from integrations.arxiv.client import arxiv_tools_enabled
from integrations.search.arxiv import _ENDPOINT, SOURCE, parse_arxiv_atom
from integrations.search.http import fetch_text, url_with_query

ARXIV_SEARCH = "arxiv_search"
ARXIV_GET_PAPER = "arxiv_get_paper"
ARXIV_TOOL_NAMES = frozenset({ARXIV_SEARCH, ARXIV_GET_PAPER})

ARXIV_TOOL_SYSTEM_HINT = (
    "Use arxiv_search to find preprints on arXiv by topic, title or author "
    "(prefixes ti:, au:, abs:, cat: are supported). Use arxiv_get_paper with an arXiv id "
    "such as 2306.04338 for one paper's abstract and PDF link. Use zotero_* for the user's own library."
)

ARXIV_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ARXIV_SEARCH,
        "description": "Search arXiv for papers. Returns id, title, authors, year, abstract and URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search phrase, or arXiv syntax like 'au:Bengio' or 'ti:attention'.",
                },
                "limit": {"type": "integer", "description": "Max hits (default 10, max 50)."},
            },
            "required": ["query"],
        },
    },
}

ARXIV_GET_PAPER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ARXIV_GET_PAPER,
        "description": "Fetch one arXiv paper by id: title, authors, abstract, abs and PDF links.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "arXiv id like '2306.04338' or '2306.04338v1', or an arxiv.org URL.",
                },
            },
            "required": ["paper_id"],
        },
    },
}


def arxiv_chat_tools() -> list[dict[str, Any]]:
    if not arxiv_tools_enabled():
        return []
    return [ARXIV_SEARCH_TOOL, ARXIV_GET_PAPER_TOOL]


def _hit_payload(hit) -> dict[str, Any]:
    arxiv_id = hit.extras.get("id", "")
    return {
        "id": arxiv_id,
        "title": hit.title,
        "authors": hit.extras.get("authors", ""),
        "year": hit.extras.get("year", ""),
        "abstract": hit.snippet,
        "url": hit.url,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
    }


def _normalize_id(raw: str) -> str:
    value = raw.strip()
    for marker in ("/abs/", "/pdf/"):
        if marker in value:
            value = value.split(marker, 1)[1]
    return value.removesuffix(".pdf").strip("/")


def execute_arxiv_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == ARXIV_SEARCH:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "error: query is required"
        try:
            limit = max(1, min(int(arguments.get("limit") or 10), 50))
        except (TypeError, ValueError):
            limit = 10
        hits = SOURCE.search(query, limit)
        return json.dumps({"count": len(hits), "papers": [_hit_payload(h) for h in hits]}, ensure_ascii=False)
    if name == ARXIV_GET_PAPER:
        paper_id = _normalize_id(str(arguments.get("paper_id") or ""))
        if not paper_id:
            return "error: paper_id is required"
        xml_text = fetch_text(
            url_with_query(_ENDPOINT, {"id_list": paper_id, "max_results": 1}),
            source=SOURCE.name,
            min_interval_s=SOURCE.min_interval_s,
            headers={"Accept": "application/atom+xml, application/xml, text/xml, */*"},
        )
        hits = parse_arxiv_atom(xml_text, limit=1)
        if not hits:
            return f"error: no arXiv paper found for id {paper_id!r}"
        return json.dumps(_hit_payload(hits[0]), ensure_ascii=False)
    return f"error: unknown arxiv tool {name!r}"
