from __future__ import annotations

import os
import time
from typing import Any

from cli.tui.tiles.base import BaseTile, TileState, wrap_fetch
from cli.tui.tiles.yahoo import fetch_yahoo_many

_LABELS = {
    "GC=F": "GOLD",
    "SI=F": "SILVER",
    "PL=F": "PLATINUM",
    "HG=F": "COPPER",
    "PA=F": "PALLADIUM",
}


def _symbols() -> list[str]:
    raw = os.environ.get("ORODRUIN_METALS_SYMBOLS", "GC=F,SI=F").strip()
    parts = [part.strip().upper() for part in raw.split(",") if part.strip()]
    return parts or ["GC=F", "SI=F"]


def _label(symbol: str) -> str:
    return _LABELS.get(symbol, symbol)


def _fetch_metals() -> TileState:
    symbols = _symbols()
    rows, errors = fetch_yahoo_many(symbols)
    lines = ["metals"]
    labeled: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row["symbol"])
        close = float(row["close"])
        change = row.get("change_pct")
        name = _label(symbol)
        if change is None:
            lines.append(f"{name}  {close:.2f}")
        else:
            sign = "+" if float(change) >= 0 else ""
            lines.append(f"{name}  {close:.2f}  ({sign}{float(change):.2f}%)")
        labeled.append(
            {
                "symbol": symbol,
                "label": name,
                "close": close,
                "change_pct": change,
            }
        )
    if errors and not labeled:
        raise ValueError("; ".join(errors))
    if errors:
        lines.append(f"[partial] {errors[0]}")
    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=time.time(),
        payload={"symbols": labeled, "provider": "yahoo"},
        error="; ".join(errors) if errors else "",
    )


class MetalsTile(BaseTile):
    tile_id = "metals"
    refresh_s = 600.0

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "metals", _fetch_metals)
