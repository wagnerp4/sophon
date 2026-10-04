from __future__ import annotations

import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_DEFAULT_TIMEOUT_S = 30.0
_last_request_at: dict[str, float] = {}
_interval_lock = threading.Lock()


def search_contact() -> str:
    return os.environ.get("SOPHON_SEARCH_CONTACT", "").strip()


def set_search_contact(value: str) -> str:
    text = value.strip()
    if not text or text.lower() in {"clear", "none", "off"}:
        os.environ.pop("SOPHON_SEARCH_CONTACT", None)
        return ""
    os.environ["SOPHON_SEARCH_CONTACT"] = text
    return text


def user_agent() -> str:
    contact = search_contact()
    if contact:
        return f"Sophon/0.1 (search; {contact}; +https://github.com/wagnerp4/sophon)"
    return "Sophon/0.1 (search; +https://github.com/wagnerp4/sophon)"


def wait_interval(source: str, min_interval_s: float) -> None:
    if min_interval_s <= 0:
        return
    with _interval_lock:
        now = time.monotonic()
        last = _last_request_at.get(source, 0.0)
        delay = min_interval_s - (now - last)
        if delay > 0:
            time.sleep(delay)
        _last_request_at[source] = time.monotonic()


def _decode_body(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def fetch_bytes(
    url: str,
    *,
    source: str,
    min_interval_s: float = 0.0,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    headers: dict[str, str] | None = None,
) -> bytes:
    wait_interval(source, min_interval_s)
    req_headers = {
        "User-Agent": user_agent(),
        "Accept": "application/json, application/atom+xml, application/xml, text/xml, */*",
        "Accept-Encoding": "gzip",
    }
    if headers:
        req_headers.update(headers)
    attempts = 3 if source in {"arxiv", "semanticscholar"} else 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                encoding = str(response.headers.get("Content-Encoding") or "").lower()
                raw = response.read()
                if "gzip" in encoding:
                    raw = gzip.decompress(raw)
                else:
                    raw = _decode_body(raw)
                return raw
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = _decode_body(exc.read()).decode("utf-8", errors="replace")[:400]
            except OSError:
                body = ""
            if exc.code in {429, 503} and attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
                last_error = exc
                continue
            detail = f" HTTP {exc.code}"
            if body:
                detail += f": {body}"
            raise RuntimeError(f"Search request failed for {source}{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Search request failed for {source}: {exc.reason}") from exc
    raise RuntimeError(f"Search request failed for {source}") from last_error


def fetch_text(
    url: str,
    *,
    source: str,
    min_interval_s: float = 0.0,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    headers: dict[str, str] | None = None,
) -> str:
    return fetch_bytes(
        url,
        source=source,
        min_interval_s=min_interval_s,
        timeout_s=timeout_s,
        headers=headers,
    ).decode("utf-8", errors="replace")


def fetch_json(
    url: str,
    *,
    source: str,
    min_interval_s: float = 0.0,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    headers: dict[str, str] | None = None,
) -> Any:
    text = fetch_text(
        url,
        source=source,
        min_interval_s=min_interval_s,
        timeout_s=timeout_s,
        headers=headers,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Search source {source} returned invalid JSON") from exc


def url_with_query(base: str, params: dict[str, str | int | None]) -> str:
    filtered: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        filtered[key] = text
    query = urllib.parse.urlencode(filtered)
    if not query:
        return base
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}{query}"
