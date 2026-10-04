from __future__ import annotations

import re

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_ENDPOINT = "https://api.github.com/search/repositories"
_FILLER = {"latest", "recent", "new", "github", "repo", "repository", "please"}
_YEAR = re.compile(r"^20\d{2}$")


class GithubSource(SearchSource):
    name = "github"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = fetch_json(
            url_with_query(
                _ENDPOINT,
                {
                    "q": prepare_github_query(query),
                    "per_page": limit,
                    "sort": "updated",
                    "order": "desc",
                },
            ),
            source=self.name,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        return parse_github_repos(payload, limit=limit)


def prepare_github_query(query: str) -> str:
    raw = [token for token in query.replace(",", " ").split() if token]
    has_filler = any(token.lower().strip("\"'") in _FILLER for token in raw)
    if not has_filler:
        return query.strip()
    kept: list[str] = []
    for token in raw:
        low = token.lower().strip("\"'")
        if low in _FILLER:
            continue
        if _YEAR.match(low):
            continue
        kept.append(token)
    return " ".join(kept) or query.strip()


def parse_github_repos(payload: object, *, limit: int) -> list[SearchHit]:
    rows = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    hits: list[SearchHit] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        url = str(row.get("html_url") or "").strip()
        extras: dict[str, str] = {}
        full_name = str(row.get("full_name") or "").strip()
        if full_name:
            extras["id"] = full_name
        license_row = row.get("license")
        if isinstance(license_row, dict):
            spdx = str(license_row.get("spdx_id") or "").strip()
            if spdx and spdx != "NOASSERTION":
                extras["license"] = spdx
        if url:
            hits.append(
                SearchHit(
                    title=full_name or str(row.get("name") or "").strip() or "(untitled)",
                    url=url,
                    snippet=str(row.get("description") or "").strip(),
                    extras=extras,
                )
            )
    return hits


SOURCE = GithubSource()
