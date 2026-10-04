from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

_API = "https://api.github.com"
_ACCEPT = "application/vnd.github+json"


def github_token() -> str:
    for key in ("SOPHON_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    return ""


def github_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_GITHUB_TOOLS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def github_request(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any, str]:
    token = github_token()
    if not token:
        return 0, None, "error: set SOPHON_GITHUB_TOKEN (or GH_TOKEN) in .env"
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        _API + path,
        data=data,
        method=method,
        headers={
            "Accept": _ACCEPT,
            "Authorization": "Bearer " + token,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Sophon/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
            code = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        code = int(exc.code)
    except Exception as exc:
        return 0, None, f"error: {exc}"
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        payload = raw
    if code >= 400:
        message = ""
        if isinstance(payload, dict):
            message = str(payload.get("message") or "")
        return code, payload, f"error: github {code} {message}".strip()
    return code, payload, ""
