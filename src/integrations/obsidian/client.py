from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


_DEFAULT_URL = "https://127.0.0.1:27124"
_MAX_BODY_CHARS = 12_000


def obsidian_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_OBSIDIAN_TOOLS", "0").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def obsidian_api_url() -> str:
    raw = os.environ.get("SOPHON_OBSIDIAN_API_URL", "").strip()
    if raw:
        return raw.rstrip("/")
    return _DEFAULT_URL


def obsidian_api_key() -> str | None:
    for key in ("SOPHON_OBSIDIAN_API_KEY", "OBSIDIAN_API_KEY", "OBSIDIAN_API_TOKEN"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    return None


def _ssl_context(url: str) -> ssl.SSLContext | None:
    if not url.lower().startswith("https://"):
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _auth_headers() -> dict[str, str]:
    headers = {"User-Agent": "sophon-obsidian/0.1", "Accept": "*/*"}
    key = obsidian_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _request(
    method: str,
    path: str,
    *,
    query: dict[str, str] | None = None,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout_s: float = 30.0,
    expect_json: bool = True,
) -> Any:
    base = obsidian_api_url()
    url = f"{base}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    headers = _auth_headers()
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    ctx = _ssl_context(base)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            ctype = (resp.headers.get("Content-Type") or "").lower()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Obsidian HTTP {exc.code} on {path}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Obsidian unreachable at {base} ({exc.reason}). "
            "Start Obsidian with Local REST API enabled."
        ) from exc
    if not raw.strip():
        return {} if expect_json else ""
    if expect_json or "application/json" in ctype:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if expect_json:
                raise
            return raw
    return raw


def _truncate(text: str, limit: int = _MAX_BODY_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


@dataclass
class ObsidianClient:
    timeout_s: float = 30.0

    def ping(self) -> dict[str, Any]:
        data = _request("GET", "/", timeout_s=min(self.timeout_s, 5.0), expect_json=True)
        return data if isinstance(data, dict) else {"ok": True, "raw": str(data)}

    def list_dir(self, path: str = "") -> list[str]:
        rel = path.strip().lstrip("/")
        if rel and not rel.endswith("/"):
            rel = rel + "/"
        api_path = f"/vault/{rel}" if rel else "/vault/"
        data = _request("GET", api_path, timeout_s=self.timeout_s, expect_json=True)
        if isinstance(data, dict):
            files = data.get("files")
            if isinstance(files, list):
                return [str(x) for x in files]
        return []

    def get_file(self, path: str) -> str:
        rel = path.strip().lstrip("/")
        if not rel:
            raise ValueError("empty vault path")
        data = _request(
            "GET",
            f"/vault/{urllib.parse.quote(rel, safe='/')}",
            timeout_s=self.timeout_s,
            expect_json=False,
        )
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2)
        return _truncate(text)

    def search_simple(self, query: str) -> Any:
        q = query.strip()
        if not q:
            raise ValueError("empty search query")
        data = _request(
            "POST",
            "/search/simple/",
            query={"query": q},
            body=b"",
            timeout_s=self.timeout_s,
            expect_json=True,
        )
        return data

    def recent_proxy(self, limit: int = 10) -> list[str]:
        n = max(1, min(int(limit), 100))
        files = self.list_dir("")
        return files[:n]


def format_tool_payload(data: Any) -> str:
    if isinstance(data, str):
        return _truncate(data)
    try:
        return _truncate(json.dumps(data, ensure_ascii=False, indent=2))
    except TypeError:
        return _truncate(str(data))
