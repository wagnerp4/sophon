from __future__ import annotations

import csv
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote

from cli.tui.tiles.base import BaseTile, TileState, http_get_text, wrap_fetch
from cli.tui.tiles.yahoo import fetch_yahoo_many


def _symbols() -> list[str]:
    raw = os.environ.get("SOPHON_QUOTES_SYMBOLS", "AAPL,MSFT,NVDA").strip()
    parts = [part.strip().upper() for part in raw.split(",") if part.strip()]
    return parts or ["AAPL", "MSFT", "NVDA"]


def _stooq_symbol(symbol: str) -> str:
    if "." in symbol:
        return symbol.lower()
    return f"{symbol.lower()}.us"


def _fetch_one_stooq(symbol: str) -> tuple[str, float, float | None]:
    stooq = _stooq_symbol(symbol)
    url = f"https://stooq.com/q/l/?s={quote(stooq)}&f=sd2t2ohlcv&h&e=csv"
    text = http_get_text(url)
    reader = csv.DictReader(io.StringIO(text))
    row = next(reader, None)
    if row is None:
        raise ValueError(f"empty quote for {symbol}")
    close_raw = (row.get("Close") or row.get("close") or "").strip()
    if not close_raw or close_raw.upper() == "N/D":
        raise ValueError(f"no price for {symbol}")
    close = float(close_raw)
    open_raw = (row.get("Open") or row.get("open") or "").strip()
    change = None
    if open_raw and open_raw.upper() != "N/D":
        open_px = float(open_raw)
        if open_px:
            change = ((close - open_px) / open_px) * 100.0
    return symbol, close, change


def _format_rows(title: str, rows: list[dict[str, Any]], errors: list[str], provider: str) -> TileState:
    lines = [title]
    for row in rows:
        sym = str(row["symbol"])
        close = float(row["close"])
        change = row.get("change_pct")
        if change is None:
            lines.append(f"{sym}  {close:.2f}")
        else:
            sign = "+" if float(change) >= 0 else ""
            lines.append(f"{sym}  {close:.2f}  ({sign}{float(change):.2f}%)")
    if errors and not rows:
        raise ValueError("; ".join(errors))
    if errors:
        lines.append(f"[partial] {errors[0]}")
    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=time.time(),
        payload={"symbols": rows, "provider": provider},
        error="; ".join(errors) if errors else "",
    )


def _fetch_quotes() -> TileState:
    provider = os.environ.get("SOPHON_QUOTES_PROVIDER", "yahoo").strip().lower() or "yahoo"
    if provider not in ("yahoo", "stooq"):
        raise ValueError(f"unsupported SOPHON_QUOTES_PROVIDER={provider!r} (use yahoo or stooq)")
    symbols = _symbols()
    if provider == "yahoo":
        rows, errors = fetch_yahoo_many(symbols)
        return _format_rows("quotes", rows, errors, provider)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(6, len(symbols) or 1)) as pool:
        futures = {pool.submit(_fetch_one_stooq, symbol): symbol for symbol in symbols}
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
    return _format_rows("quotes", rows, errors, provider)


class QuotesTile(BaseTile):
    tile_id = "quotes"
    refresh_s = 600.0

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "quotes", _fetch_quotes)
