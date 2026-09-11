from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from cli.tui.tiles.base import BaseTile, TileState, http_get_json, wrap_fetch

_KRAKEN_PAIR = {
    "BTC": "XBTUSD",
    "XBT": "XBTUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
}

_KRAKEN_RESULT_ALIASES = {
    "XBTUSD": ("XXBTZUSD", "XBTUSD"),
    "ETHUSD": ("XETHZUSD", "ETHUSD"),
    "SOLUSD": ("SOLUSD",),
}


def _symbols() -> list[str]:
    raw = os.environ.get("ORODRUIN_CRYPTO_SYMBOLS", "BTC,ETH,SOL").strip()
    parts = [part.strip().upper() for part in raw.split(",") if part.strip()]
    return parts or ["BTC", "ETH", "SOL"]


def _fmt_price(price: float) -> str:
    if price >= 100:
        return f"{price:.2f}"
    if price >= 1:
        return f"{price:.2f}"
    if price >= 0.01:
        return f"{price:.4f}"
    return f"{price:.6f}"


def _kraken_pair(symbol: str) -> str:
    return _KRAKEN_PAIR.get(symbol, f"{symbol}USD")


def _fetch_kraken() -> TileState:
    symbols = _symbols()
    pairs = [_kraken_pair(symbol) for symbol in symbols]
    url = f"https://api.kraken.com/0/public/Ticker?pair={quote(','.join(pairs))}"
    data = http_get_json(url)
    result = (data or {}).get("result") or {}
    if not isinstance(result, dict) or not result:
        raise ValueError("kraken ticker empty")
    rows: list[dict[str, Any]] = []
    lines = ["crypto"]
    for symbol, pair in zip(symbols, pairs):
        entry = None
        for key in _KRAKEN_RESULT_ALIASES.get(pair, (pair,)):
            if key in result:
                entry = result[key]
                break
        if entry is None:
            for key, value in result.items():
                if pair[:3] in key or symbol in key:
                    entry = value
                    break
        if not isinstance(entry, dict):
            lines.append(f"{symbol}  —")
            continue
        last = float((entry.get("c") or ["0"])[0])
        open_px = float(entry.get("o") or 0)
        change = ((last - open_px) / open_px) * 100.0 if open_px else None
        if change is None:
            lines.append(f"{symbol}  {_fmt_price(last)}")
        else:
            sign = "+" if change >= 0 else ""
            lines.append(f"{symbol}  {_fmt_price(last)}  ({sign}{change:.2f}%)")
        rows.append(
            {
                "symbol": symbol,
                "close": last,
                "change_pct": change,
            }
        )
    if not rows:
        raise ValueError("kraken returned no usable pairs")
    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=time_module(),
        payload={"symbols": rows, "provider": "kraken"},
    )


def time_module() -> float:
    import time

    return time.time()


def _fetch_binance() -> TileState:
    symbols = _symbols()
    tickers = [f"{symbol}USDT" for symbol in symbols]
    encoded = quote(str(tickers).replace("'", '"'))
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbols={encoded}"
    data = http_get_json(url)
    if not isinstance(data, list):
        raise ValueError("binance ticker empty")
    by_symbol = {str(item.get("symbol")): item for item in data if isinstance(item, dict)}
    rows: list[dict[str, Any]] = []
    lines = ["crypto"]
    for symbol, ticker in zip(symbols, tickers):
        item = by_symbol.get(ticker)
        if item is None:
            lines.append(f"{symbol}  —")
            continue
        last = float(item.get("lastPrice") or 0)
        change = float(item.get("priceChangePercent") or 0)
        sign = "+" if change >= 0 else ""
        lines.append(f"{symbol}  {_fmt_price(last)}  ({sign}{change:.2f}%)")
        rows.append(
            {
                "symbol": symbol,
                "close": last,
                "change_pct": change,
            }
        )
    if not rows:
        raise ValueError("binance returned no usable pairs")
    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=time_module(),
        payload={"symbols": rows, "provider": "binance"},
    )


def _fetch_crypto() -> TileState:
    provider = os.environ.get("ORODRUIN_CRYPTO_PROVIDER", "kraken").strip().lower() or "kraken"
    if provider == "binance":
        return _fetch_binance()
    if provider != "kraken":
        raise ValueError(f"unsupported ORODRUIN_CRYPTO_PROVIDER={provider!r} (use kraken or binance)")
    return _fetch_kraken()


class CryptoTilePanel(Vertical):
    DEFAULT_CSS = """
    CryptoTilePanel {
        border: solid $primary;
        padding: 0 1;
        height: 1fr;
        min-height: 8;
    }
    #crypto-title {
        height: 1;
        text-style: bold;
    }
    #crypto-rows {
        height: 1fr;
    }
    .crypto-line {
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("crypto", id="crypto-title")
        yield Vertical(id="crypto-rows")

    def apply(self, state: TileState) -> None:
        title = self.query_one("#crypto-title", Static)
        title.update("crypto" if state.ok else "crypto [error]")
        rows_box = self.query_one("#crypto-rows", Vertical)
        for child in list(rows_box.children):
            child.remove()
        payload = state.payload or {}
        symbols = payload.get("symbols") if isinstance(payload, dict) else None
        if not isinstance(symbols, list) or not symbols:
            rows_box.mount(Static(state.text))
            return
        for row in symbols:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "?")
            close = row.get("close")
            change = row.get("change_pct")
            if change is None:
                label = f"{symbol}  {_fmt_price(float(close))}"
            else:
                sign = "+" if float(change) >= 0 else ""
                label = f"{symbol}  {_fmt_price(float(close))}  ({sign}{float(change):.2f}%)"
            rows_box.mount(Static(label, classes="crypto-line"))


class CryptoTile(BaseTile):
    tile_id = "crypto"
    refresh_s = 60.0

    def panel(self) -> Any:
        return CryptoTilePanel(id=f"dash-{self.tile_id}")

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "crypto", _fetch_crypto)
