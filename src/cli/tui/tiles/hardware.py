from __future__ import annotations

import base64
import json
import os
import statistics
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from cli.tui.tiles.base import (
    BaseTile,
    TileState,
    dashboard_cache_dir,
    http_get_json,
    load_cached_state,
    wrap_fetch,
)


def _queries() -> list[str]:
    raw = os.environ.get(
        "SOPHON_HARDWARE_QUERIES",
        "RTX 4090,RTX 3090,Ryzen 9 7950X",
    ).strip()
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts or ["RTX 4090", "RTX 3090"]


def _credentials() -> tuple[str, str] | None:
    client_id = os.environ.get("SOPHON_EBAY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SOPHON_EBAY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    return client_id, client_secret


def _token_cache_path():
    return dashboard_cache_dir() / "ebay_token.json"


def _load_cached_token() -> str | None:
    path = _token_cache_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    token = str(raw.get("access_token") or "")
    expires_at = float(raw.get("expires_at") or 0)
    if not token or time.time() >= expires_at - 60:
        return None
    return token


def _save_cached_token(token: str, expires_in: int) -> None:
    path = _token_cache_path()
    payload = {
        "access_token": token,
        "expires_at": time.time() + max(60, int(expires_in)),
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _fetch_ebay_token(client_id: str, client_secret: str) -> str:
    cached = _load_cached_token()
    if cached:
        return cached
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    body = urlencode(
        {
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        }
    ).encode("utf-8")
    req = Request(
        "https://api.ebay.com/identity/v1/oauth2/token",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {basic}",
            "User-Agent": "sophon-dashboard/0.1",
        },
        method="POST",
    )
    with urlopen(req, timeout=12.0) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    token = str(data.get("access_token") or "")
    if not token:
        raise ValueError("ebay token response missing access_token")
    _save_cached_token(token, int(data.get("expires_in") or 7200))
    return token


def _search_query(token: str, query: str) -> dict[str, Any]:
    params = urlencode(
        {
            "q": query,
            "limit": "50",
            "filter": "conditions:{USED}",
        }
    )
    url = f"https://api.ebay.com/buy/browse/v1/item_summary/search?{params}"
    data = http_get_json(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_DE",
        },
    )
    items = data.get("itemSummaries") if isinstance(data, dict) else None
    prices: list[float] = []
    currency = "EUR"
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            price = item.get("price") or {}
            value = price.get("value")
            if value is None:
                continue
            try:
                prices.append(float(value))
            except (TypeError, ValueError):
                continue
            currency = str(price.get("currency") or currency)
    if not prices:
        return {
            "query": query,
            "count": 0,
            "median": None,
            "minimum": None,
            "currency": currency,
        }
    return {
        "query": query,
        "count": len(prices),
        "median": float(statistics.median(prices)),
        "minimum": float(min(prices)),
        "currency": currency,
    }


def _fetch_hardware() -> TileState:
    creds = _credentials()
    if creds is None:
        return TileState(
            ok=False,
            text=(
                "hardware\n"
                "set SOPHON_EBAY_CLIENT_ID / SOPHON_EBAY_CLIENT_SECRET\n"
                "(asking prices via eBay Browse; sold history not available)"
            ),
            fetched_at=time.time(),
            error="missing ebay credentials",
        )
    client_id, client_secret = creds
    token = _fetch_ebay_token(client_id, client_secret)
    rows: list[dict[str, Any]] = []
    lines = ["hardware (asking)"]
    errors: list[str] = []
    for query in _queries():
        try:
            row = _search_query(token, query)
        except Exception as exc:
            errors.append(f"{query}: {exc}")
            lines.append(f"{query}  —")
            continue
        rows.append(row)
        if not row["count"]:
            lines.append(f"{query}  no used listings")
            continue
        lines.append(
            f"{query}  med {row['median']:.0f} {row['currency']}  "
            f"min {row['minimum']:.0f}  n={row['count']}"
        )
    if errors and not rows:
        raise ValueError("; ".join(errors))
    if errors:
        lines.append(f"[partial] {errors[0]}")
    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=time.time(),
        payload={"rows": rows, "provider": "ebay_browse"},
        error="; ".join(errors) if errors else "",
    )


HARDWARE_TTL_S = 21600.0


def fetch_hardware_state(*, force: bool = False) -> TileState:
    if not force:
        cached = load_cached_state("hardware")
        if cached is not None and (time.time() - cached.fetched_at) < HARDWARE_TTL_S:
            return cached
    return wrap_fetch("hardware", "hardware", _fetch_hardware)


class HardwareTile(BaseTile):
    tile_id = "hardware"
    refresh_s = HARDWARE_TTL_S

    def fetch(self, app: Any) -> TileState:
        return fetch_hardware_state()
