from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import quote

from cli.tui.tiles.base import BaseTile, TileState, http_get_json, wrap_fetch


def _limit() -> int:
    raw = os.environ.get("ORODRUIN_POLYMARKET_LIMIT", "5").strip()
    try:
        return max(1, min(int(raw or "5"), 15))
    except ValueError:
        return 5


def _slugs() -> list[str]:
    raw = os.environ.get("ORODRUIN_POLYMARKET_MARKETS", "").strip()
    return [part.strip() for part in raw.split(",") if part.strip()]


def _pct(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if num <= 1.0:
        num *= 100.0
    return f"{num:.1f}%"


def _short(text: str, width: int = 56) -> str:
    clean = " ".join(text.split())
    if len(clean) <= width:
        return clean
    return clean[: width - 1] + "…"


def _market_rows_from_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    title = str(event.get("title") or event.get("slug") or "event")
    markets = event.get("markets")
    rows: list[dict[str, Any]] = []
    if isinstance(markets, list) and markets:
        for market in markets[:3]:
            if not isinstance(market, dict):
                continue
            q = str(market.get("question") or market.get("slug") or title)
            prices = market.get("outcomePrices")
            price = None
            if isinstance(prices, str):
                try:
                    parsed = json.loads(prices)
                    if isinstance(parsed, list) and parsed:
                        price = parsed[0]
                except Exception:
                    price = None
            elif isinstance(prices, list) and prices:
                price = prices[0]
            if price is None:
                price = market.get("lastTradePrice") or market.get("bestBid")
            rows.append(
                {
                    "title": q,
                    "price": price,
                    "slug": str(event.get("slug") or ""),
                    "volume": market.get("volume24hr") or event.get("volume24hr"),
                }
            )
    else:
        rows.append(
            {
                "title": title,
                "price": event.get("lastTradePrice"),
                "slug": str(event.get("slug") or ""),
                "volume": event.get("volume24hr"),
            }
        )
    return rows


def _fetch_by_slugs(slugs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slug in slugs:
        url = f"https://gamma-api.polymarket.com/events?slug={quote(slug)}"
        data = http_get_json(url, timeout_s=10.0)
        events = data if isinstance(data, list) else []
        if not events and isinstance(data, dict):
            events = [data]
        for event in events:
            if isinstance(event, dict):
                rows.extend(_market_rows_from_event(event))
    return rows


def _fetch_top(limit: int) -> list[dict[str, Any]]:
    url = (
        "https://gamma-api.polymarket.com/events"
        f"?limit={limit}&active=true&closed=false&order=volume24hr&ascending=false"
    )
    data = http_get_json(url, timeout_s=12.0)
    if not isinstance(data, list):
        raise ValueError("polymarket events empty")
    rows: list[dict[str, Any]] = []
    for event in data:
        if isinstance(event, dict):
            rows.extend(_market_rows_from_event(event)[:1])
        if len(rows) >= limit:
            break
    return rows[:limit]


def _fetch_polymarket() -> TileState:
    limit = _limit()
    slugs = _slugs()
    if slugs:
        rows = _fetch_by_slugs(slugs)[:limit]
    else:
        rows = _fetch_top(limit)
    if not rows:
        raise ValueError("no polymarket markets returned")
    lines = ["polymarket"]
    for row in rows:
        lines.append(f"{_pct(row.get('price'))}  {_short(str(row.get('title') or ''))}")
    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=time.time(),
        payload={"markets": rows, "mode": "slugs" if slugs else "top"},
    )


class PolymarketTile(BaseTile):
    tile_id = "polymarket"
    refresh_s = 120.0

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "polymarket", _fetch_polymarket)
