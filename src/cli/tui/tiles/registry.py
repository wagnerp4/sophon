from __future__ import annotations

import os
from typing import Any

from cli.tui.tiles.crypto import CryptoTile
from cli.tui.tiles.hardware import HardwareTile
from cli.tui.tiles.host import HostTile
from cli.tui.tiles.markets import MarketsTile
from cli.tui.tiles.metals import MetalsTile
from cli.tui.tiles.news import NewsTile
from cli.tui.tiles.polymarket import PolymarketTile
from cli.tui.tiles.quotes import QuotesTile
from cli.tui.tiles.services import ServicesTile
from cli.tui.tiles.storage import StorageTile
from cli.tui.tiles.system import SystemTile
from cli.tui.tiles.twitch import TwitchTile
from cli.tui.tiles.vault import VaultTile
from cli.tui.tiles.weather import WeatherTile

_BUILTIN: dict[str, Any] = {
    "host": HostTile,
    "weather": WeatherTile,
    "system": SystemTile,
    "gpu": SystemTile,
    "storage": StorageTile,
    "markets": MarketsTile,
    "crypto": CryptoTile,
    "metals": MetalsTile,
    "quotes": QuotesTile,
    "hardware": HardwareTile,
    "news": NewsTile,
    "vault": VaultTile,
    "services": ServicesTile,
    "polymarket": PolymarketTile,
    "twitch": TwitchTile,
}

DEFAULT_TILES = (
    "host",
    "weather",
    "services",
    "storage",
    "system",
    "markets",
    "news",
    "polymarket",
    "twitch",
)


def configured_tile_ids() -> list[str]:
    raw = os.environ.get("SOPHON_DASHBOARD_TILES", "").strip()
    if not raw:
        return list(DEFAULT_TILES)
    ids = [part.strip().lower() for part in raw.split(",") if part.strip()]
    normalized: list[str] = []
    seen: set[str] = set()
    for tile_id in ids:
        if tile_id == "gpu":
            tile_id = "system"
        if tile_id == "hardware" and "host" in ids:
            continue
        if tile_id in seen:
            continue
        seen.add(tile_id)
        normalized.append(tile_id)
    return normalized


def build_dashboard_tiles() -> tuple[list[Any], list[str]]:
    tiles: list[Any] = []
    unknown: list[str] = []
    for tile_id in configured_tile_ids():
        cls = _BUILTIN.get(tile_id)
        if cls is None:
            unknown.append(tile_id)
            continue
        tiles.append(cls())
    if not tiles:
        tiles.append(HostTile())
    return tiles, unknown
