from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from cli.tui.tiles.base import BaseTile, TileState, wrap_fetch
from cli.tui.tiles.crypto import _fetch_crypto
from cli.tui.tiles.crypto import _fmt_price as _fmt_crypto_price
from cli.tui.tiles.metals import _fetch_metals
from cli.tui.tiles.movers import fetch_period_movers
from cli.tui.tiles.quotes import _fetch_quotes


def _fmt_price(price: Any) -> str:
    try:
        value = float(price)
    except (TypeError, ValueError):
        return "—"
    return _fmt_crypto_price(value)


def _fmt_row(symbol: str, close: Any, change: Any) -> str:
    if close is None:
        return f"{symbol}  —"
    price = _fmt_price(close)
    if change is None:
        return f"{symbol}  {price}"
    try:
        pct = float(change)
    except (TypeError, ValueError):
        return f"{symbol}  {price}"
    sign = "+" if pct >= 0 else ""
    return f"{symbol}  {price}  ({sign}{pct:.2f}%)"


def _fmt_mover(row: dict[str, Any]) -> str:
    horizon = str(row.get("horizon") or "?")
    label = str(row.get("label") or "—")
    asset = str(row.get("asset_class") or "")
    change = row.get("change_pct")
    if change is None:
        return f"{horizon}  {label}"
    sign = "+" if float(change) >= 0 else ""
    asset_bit = f" · {asset}" if asset else ""
    return f"{horizon}  {label}  ({sign}{float(change):.2f}%){asset_bit}"


def _section_rows(state: TileState, *, label_key: str = "symbol") -> list[dict[str, Any]]:
    payload = state.payload or {}
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(symbols, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in symbols:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "label": str(row.get("label") or row.get(label_key) or "?"),
                "close": row.get("close"),
                "change_pct": row.get("change_pct"),
            }
        )
    return rows


def _fetch_markets() -> TileState:
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_crypto = pool.submit(_fetch_crypto)
        fut_metals = pool.submit(_fetch_metals)
        fut_quotes = pool.submit(_fetch_quotes)
        fut_movers = pool.submit(fetch_period_movers)
        crypto = fut_crypto.result()
        metals = fut_metals.result()
        quotes = fut_quotes.result()
        try:
            movers = fut_movers.result()
            movers_error = ""
        except Exception as exc:
            movers = []
            movers_error = str(exc)

    sections = {
        "crypto": _section_rows(crypto),
        "metals": _section_rows(metals),
        "quotes": _section_rows(quotes),
        "movers": movers if isinstance(movers, list) else [],
    }
    lines = ["markets"]
    for name in ("crypto", "metals", "quotes", "movers"):
        rows = sections[name]
        lines.append(f"[{name}]")
        if not rows:
            if name == "movers":
                lines.append(f"  {movers_error or 'unavailable'}")
            else:
                src = crypto if name == "crypto" else metals if name == "metals" else quotes
                lines.append(f"  {src.error or 'unavailable'}")
            continue
        for row in rows:
            if name == "movers":
                lines.append(f"  {_fmt_mover(row)}")
            else:
                lines.append(f"  {_fmt_row(row['label'], row['close'], row['change_pct'])}")

    ok = crypto.ok or metals.ok or quotes.ok or bool(sections["movers"])
    errors = [part for part in (crypto.error, metals.error, quotes.error, movers_error) if part]
    return TileState(
        ok=ok,
        text="\n".join(lines),
        fetched_at=time.time(),
        payload={
            "crypto": sections["crypto"],
            "metals": sections["metals"],
            "quotes": sections["quotes"],
            "movers": sections["movers"],
        },
        error="; ".join(errors),
    )


class MarketsTilePanel(Vertical):
    DEFAULT_CSS = """
    MarketsTilePanel {
        border: solid $primary;
        padding: 0 1;
        height: 1fr;
        min-height: 10;
    }
    MarketsTilePanel.tile-span-2 {
        column-span: 2;
    }
    #markets-title {
        height: 1;
        text-style: bold;
    }
    #markets-body {
        height: 1fr;
        layout: horizontal;
    }
    .markets-col {
        width: 1fr;
        height: 1fr;
        padding: 0 1 0 0;
    }
    .markets-col-title {
        height: 1;
        color: $text-muted;
        text-style: bold;
    }
    .markets-line {
        height: 1;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("markets", id="markets-title")
        with Horizontal(id="markets-body"):
            yield Vertical(id="markets-crypto", classes="markets-col")
            yield Vertical(id="markets-metals", classes="markets-col")
            yield Vertical(id="markets-quotes", classes="markets-col")
            yield Vertical(id="markets-movers", classes="markets-col")

    def _fill_col(self, col_id: str, title: str, rows: list[dict[str, Any]], *, movers: bool = False) -> None:
        col = self.query_one(col_id, Vertical)
        for child in list(col.children):
            child.remove()
        col.mount(Static(title, classes="markets-col-title"))
        if not rows:
            col.mount(Static("  —", classes="markets-line"))
            return
        for row in rows:
            if movers:
                text = _fmt_mover(row)
            else:
                text = _fmt_row(row["label"], row["close"], row["change_pct"])
            col.mount(Static(text, classes="markets-line"))

    def apply(self, state: TileState) -> None:
        title = self.query_one("#markets-title", Static)
        title.update("markets" if state.ok else "markets [partial]")
        payload = state.payload or {}
        self._fill_col("#markets-crypto", "crypto", list(payload.get("crypto") or []))
        self._fill_col("#markets-metals", "metals", list(payload.get("metals") or []))
        self._fill_col("#markets-quotes", "quotes", list(payload.get("quotes") or []))
        self._fill_col(
            "#markets-movers",
            "movers",
            list(payload.get("movers") or []),
            movers=True,
        )


class MarketsTile(BaseTile):
    tile_id = "markets"
    refresh_s = 60.0
    span_cols = 2

    def panel(self) -> Any:
        panel = MarketsTilePanel(id=f"dash-{self.tile_id}")
        panel.add_class("tile-span-2")
        return panel

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "markets", _fetch_markets)
