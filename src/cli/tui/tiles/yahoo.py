from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote

from cli.tui.tiles.base import http_get_json


def fetch_one_yahoo(symbol: str) -> tuple[str, float, float | None]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}"
        "?range=5d&interval=1d"
    )
    data = http_get_json(url)
    result = (((data or {}).get("chart") or {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        raise ValueError(f"empty quote for {symbol}")
    meta = result.get("meta") or {}
    close = meta.get("regularMarketPrice")
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    if close is None:
        quotes = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quotes.get("close") or []
        closes = [c for c in closes if c is not None]
        if not closes:
            raise ValueError(f"no price for {symbol}")
        close = float(closes[-1])
        prev = float(closes[-2]) if len(closes) > 1 else None
    close_f = float(close)
    change = None
    if prev:
        change = ((close_f - float(prev)) / float(prev)) * 100.0
    return symbol, close_f, change


def fetch_yahoo_many(
    symbols: list[str],
    *,
    max_workers: int = 6,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not symbols:
        return rows, errors
    workers = min(max_workers, len(symbols))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one_yahoo, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, close, change = future.result()
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")
                continue
            rows.append({"symbol": sym, "close": close, "change_pct": change})
    order = {symbol: index for index, symbol in enumerate(symbols)}
    rows.sort(key=lambda row: order.get(str(row["symbol"]), 10_000))
    return rows, errors
