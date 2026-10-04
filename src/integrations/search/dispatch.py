from __future__ import annotations

import os
from typing import Any

from integrations.search.registry import get, names
from integrations.search.types import SearchHit

_AUTO_ERROR = (
    "Set SOPHON_SEARXNG_URL (preferred) or SOPHON_GOOGLE_CSE_KEY and SOPHON_GOOGLE_CSE_CX. "
    "Named sources such as github, zenodo, arxiv, openalex, and wikipedia work without those."
)


_DISABLED_ENV = "SOPHON_SEARCH_DISABLED"


def web_search_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_WEB_SEARCH_TOOLS", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def set_web_search_tools(enabled: bool) -> None:
    os.environ["SOPHON_WEB_SEARCH_TOOLS"] = "1" if enabled else "0"


def disabled_names() -> frozenset[str]:
    raw = os.environ.get(_DISABLED_ENV, "").strip()
    if not raw:
        return frozenset()
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def set_source_disabled(name: str, *, disabled: bool) -> None:
    key = name.strip().lower()
    if key == "auto":
        raise ValueError("Cannot disable auto. Use /search off to hide web_search.")
    get(key)
    current = set(disabled_names())
    if disabled:
        current.add(key)
    else:
        current.discard(key)
    if current:
        os.environ[_DISABLED_ENV] = ",".join(sorted(current))
    else:
        os.environ.pop(_DISABLED_ENV, None)


def clear_disabled_sources() -> None:
    os.environ.pop(_DISABLED_ENV, None)


def search_web(query: str, *, limit: int = 5, source: str = "auto") -> list[SearchHit]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    n = max(1, min(int(limit), 10))
    key = (source or "auto").strip().lower() or "auto"
    blocked = disabled_names()
    if key != "auto" and key in blocked:
        raise RuntimeError(f"Search source {key!r} is disabled. /search enable {key}")
    if key == "auto":
        return _search_auto(q, n)
    provider = get(key)
    if not provider.available():
        raise RuntimeError(f"Search source {key!r} is not configured.")
    return provider.search(q, n)


def _search_auto(query: str, limit: int) -> list[SearchHit]:
    from integrations.google.searxng import fetch_searxng_hits, searxng_configured

    blocked = disabled_names()
    errors: list[str] = []
    if searxng_configured() and "searxng" not in blocked:
        try:
            return hits_from_maps(fetch_searxng_hits(query, limit=limit))
        except Exception as exc:
            errors.append(str(exc))
    cse = get("cse")
    if cse.available() and "cse" not in blocked:
        try:
            return cse.search(query, limit)
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(" | ".join(errors))
    raise RuntimeError(_AUTO_ERROR)


def hits_from_maps(rows: list[dict[str, str]]) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for row in rows:
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        hits.append(
            SearchHit(
                title=str(row.get("title") or "").strip() or "(untitled)",
                url=url,
                snippet=str(row.get("snippet") or "").strip(),
            )
        )
    return hits


def format_search_results(hits: list[SearchHit], *, query: str) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for hit in hits:
        row: dict[str, str] = {
            "title": hit.title,
            "url": hit.url,
            "snippet": hit.snippet,
        }
        if hit.extras:
            row.update(hit.extras)
        rows.append(row)
    return {"query": query, "count": len(hits), "results": rows}


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
        if hit.extras:
            bits = [f"{key}={value}" for key, value in hit.extras.items() if value]
            if bits:
                lines.append("")
                lines.append(", ".join(bits))
        lines.append("")
    return "\n".join(lines)


def source_names_for_tool() -> list[str]:
    blocked = disabled_names()
    return ["auto", *[name for name in names() if name not in blocked]]
