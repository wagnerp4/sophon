from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_MAX_RESULTS = 10
_USER_AGENT = "Sophon/0.1 (private SearXNG client)"
# TODO: auto-start scripts/searxng when SOPHON_SEARXNG_URL is localhost and the port is closed
# TODO: refuse public SearXNG hosts for automated queries (JSON is usually off and scraping them is against instance etiquette)


def searxng_base_url() -> str:
    return os.environ.get("SOPHON_SEARXNG_URL", "").strip().rstrip("/")


def searxng_configured() -> bool:
    return bool(searxng_base_url())


def searxng_engines() -> str:
    return os.environ.get("SOPHON_SEARXNG_ENGINES", "").strip()


def searxng_categories() -> str:
    return os.environ.get("SOPHON_SEARXNG_CATEGORIES", "").strip()


def fetch_searxng_hits(query: str, *, limit: int = 5) -> list[dict[str, str]]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    base = searxng_base_url()
    if not base:
        raise RuntimeError("Set SOPHON_SEARXNG_URL to a SearXNG instance (JSON format enabled).")
    n = max(1, min(int(limit), _MAX_RESULTS))
    params: dict[str, str] = {
        "q": q,
        "format": "json",
        "pageno": "1",
    }
    engines = searxng_engines()
    if engines:
        params["engines"] = engines
    categories = searxng_categories()
    if categories:
        params["categories"] = categories
    url = f"{base}/search?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            content_type = str(response.headers.get("Content-Type") or "")
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_http_error(exc.code, body)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"SearXNG failed: {exc}") from exc
    if "json" not in content_type.lower():
        stripped = raw.lstrip()
        if stripped.startswith("<") or not stripped.startswith("{"):
            raise RuntimeError(
                "SearXNG returned HTML. Enable json under search.formats in settings.yml "
                "(public instances often disable the JSON API)."
            )
    try:
        payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("SearXNG returned non-JSON. Enable search.formats json.") from exc
    hits: list[dict[str, str]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        hits.append(
            {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "snippet": str(item.get("content") or item.get("snippet") or ""),
            }
        )
        if len(hits) >= n:
            break
    return hits


def _http_error(code: int, body: str) -> str:
    snippet = " ".join((body or "").split())[:240]
    if code == 403:
        return (
            f"SearXNG HTTP 403: JSON format is disabled on this instance. "
            f"Add json to search.formats. {snippet}"
        )
    return f"SearXNG HTTP {code}: {snippet}"
