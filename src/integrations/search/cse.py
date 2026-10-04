from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
_MAX_RESULTS = 10


def cse_api_key() -> str:
    return os.environ.get("SOPHON_GOOGLE_CSE_KEY", "").strip()


def cse_cx() -> str:
    return os.environ.get("SOPHON_GOOGLE_CSE_CX", "").strip()


def cse_configured() -> bool:
    return bool(cse_api_key() and cse_cx())


def search_cse(query: str, limit: int) -> list[SearchHit]:
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
    url = f"{_CSE_ENDPOINT}?{params}"
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


class CseSource(SearchSource):
    name = "cse"

    def available(self) -> bool:
        return cse_configured()

    def search(self, query: str, limit: int) -> list[SearchHit]:
        if not self.available():
            raise RuntimeError(
                "Set SOPHON_GOOGLE_CSE_KEY and SOPHON_GOOGLE_CSE_CX for Custom Search."
            )
        return search_cse(query, limit)


SOURCE = CseSource()
