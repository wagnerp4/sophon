from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

_MAX_RESULTS = 10


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


def cse_configured() -> bool:
    return bool(cse_api_key() and cse_cx())


def cse_api_key() -> str:
    return os.environ.get("SOPHON_GOOGLE_CSE_KEY", "").strip()


def cse_cx() -> str:
    return os.environ.get("SOPHON_GOOGLE_CSE_CX", "").strip()


def web_search_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_WEB_SEARCH_TOOLS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    from integrations.google.searxng import searxng_configured

    return searxng_configured() or cse_configured()


def search_web(query: str, *, limit: int = 5) -> list[SearchHit]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    from integrations.google.searxng import fetch_searxng_hits, searxng_configured

    errors: list[str] = []
    if searxng_configured():
        try:
            rows = fetch_searxng_hits(q, limit=limit)
            return [
                SearchHit(title=row["title"], url=row["url"], snippet=row["snippet"])
                for row in rows
            ]
        except Exception as exc:
            errors.append(str(exc))
    if cse_configured():
        try:
            return _search_cse(q, limit=limit)
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(" | ".join(errors))
    raise RuntimeError(
        "Set SOPHON_SEARXNG_URL (preferred) or SOPHON_GOOGLE_CSE_KEY and SOPHON_GOOGLE_CSE_CX."
    )


def _search_cse(query: str, *, limit: int) -> list[SearchHit]:
    key = cse_api_key()
    cx = cse_cx()
    if not key or not cx:
        raise RuntimeError(
            "Set SOPHON_GOOGLE_CSE_KEY and SOPHON_GOOGLE_CSE_CX for Custom Search."
        )
    n = max(1, min(int(limit), _MAX_RESULTS))
    params = urllib.parse.urlencode(
        {
            "key": key,
            "cx": cx,
            "q": query,
            "num": str(n),
        }
    )
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_cse_http_error(exc.code, body)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Custom Search failed: {exc}") from exc
    hits: list[SearchHit] = []
    for item in payload.get("items") or []:
        hits.append(
            SearchHit(
                title=str(item.get("title") or ""),
                url=str(item.get("link") or ""),
                snippet=str(item.get("snippet") or ""),
            )
        )
    return hits


def _cse_http_error(code: int, body: str) -> str:
    snippet = " ".join((body or "").split())[:320]
    if code == 403 and "Custom Search JSON API" in snippet:
        return (
            "Custom Search HTTP 403: this Google Cloud project cannot call the Custom Search JSON API. "
            "Enable that API on the same project as SOPHON_GOOGLE_CSE_KEY, or set SOPHON_SEARXNG_URL. "
            f"{snippet}"
        )
    return f"Custom Search HTTP {code}: {snippet}"


def format_search_results(hits: list[SearchHit], *, query: str) -> dict[str, Any]:
    return {
        "query": query,
        "count": len(hits),
        "results": [
            {"title": h.title, "url": h.url, "snippet": h.snippet}
            for h in hits
        ],
    }


def search_to_markdown(hits: list[SearchHit], *, query: str) -> str:
    lines = [f"# Web search: {query}", ""]
    if not hits:
        lines.append("_No results._")
        return "\n".join(lines)
    for idx, hit in enumerate(hits, start=1):
        lines.append(f"## {idx}. {hit.title or hit.url}")
        lines.append(hit.url)
        if hit.snippet:
            lines.append("")
            lines.append(hit.snippet)
        lines.append("")
    return "\n".join(lines)
