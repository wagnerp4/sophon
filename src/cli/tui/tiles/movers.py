from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote

from cli.tui.tiles.base import http_get_json
from cli.tui.tiles.metals import _LABELS as _METAL_LABELS
from cli.tui.tiles.metals import _symbols as _metal_symbols
from cli.tui.tiles.quotes import _symbols as _quote_symbols

_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
_CORE_CRYPTO = ("BTC", "ETH", "SOL")


def _crypto_symbols() -> list[str]:
    raw = os.environ.get("ORODRUIN_CRYPTO_SYMBOLS", "BTC,ETH,SOL").strip()
    parts = [part.strip().upper() for part in raw.split(",") if part.strip()]
    return parts or list(_CORE_CRYPTO)


def _is_leveraged_token(base: str) -> bool:
    upper = base.upper()
    return any(upper.endswith(suffix) for suffix in _LEVERAGED_SUFFIXES)


def _pct(last: float, prior: float | None) -> float | None:
    if prior is None or prior == 0:
        return None
    return ((last - prior) / prior) * 100.0


def _closes_from_binance_klines(symbol: str, *, limit: int = 35) -> list[float]:
    url = (
        "https://api.binance.com/api/v3/klines"
        f"?symbol={quote(symbol)}&interval=1d&limit={limit}"
    )
    data = http_get_json(url, timeout_s=12.0)
    if not isinstance(data, list):
        return []
    closes: list[float] = []
    for candle in data:
        if isinstance(candle, (list, tuple)) and len(candle) >= 5:
            closes.append(float(candle[4]))
    return closes


def _horizons_from_closes(closes: list[float]) -> dict[str, float | None]:
    if not closes:
        return {"close": None, "d": None, "w": None, "m": None}
    last = float(closes[-1])
    prior_d = float(closes[-2]) if len(closes) >= 2 else None
    prior_w = float(closes[-6]) if len(closes) >= 6 else None
    prior_m = float(closes[-22]) if len(closes) >= 22 else (float(closes[0]) if len(closes) >= 2 else None)
    return {
        "close": last,
        "d": _pct(last, prior_d),
        "w": _pct(last, prior_w),
        "m": _pct(last, prior_m),
    }


def _yahoo_horizons(symbol: str) -> dict[str, Any]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}"
        "?range=3mo&interval=1d"
    )
    data = http_get_json(url, timeout_s=12.0)
    result = (((data or {}).get("chart") or {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        raise ValueError(f"empty chart for {symbol}")
    quotes = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes_raw = quotes.get("close") or []
    closes = [float(value) for value in closes_raw if value is not None]
    meta = result.get("meta") or {}
    last_meta = meta.get("regularMarketPrice")
    if last_meta is not None and closes:
        closes[-1] = float(last_meta)
    elif last_meta is not None and not closes:
        closes = [float(last_meta)]
    horizons = _horizons_from_closes(closes)
    if horizons.get("d") is None:
        prev = meta.get("chartPreviousClose") or meta.get("previousClose")
        if last_meta is not None and prev:
            horizons["d"] = _pct(float(last_meta), float(prev))
            horizons["close"] = float(last_meta)
    return horizons


def _binance_daily_gainers(*, limit: int = 5, exclude: set[str] | None = None) -> list[str]:
    skip = {item.upper() for item in (exclude or set())}
    data = http_get_json("https://api.binance.com/api/v3/ticker/24hr", timeout_s=12.0)
    if not isinstance(data, list):
        return []
    ranked: list[tuple[float, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        market = str(item.get("symbol") or "")
        if not market.endswith("USDT"):
            continue
        base = market[:-4].upper()
        if not base or base in skip or _is_leveraged_token(base):
            continue
        try:
            change = float(item.get("priceChangePercent") or 0)
            last = float(item.get("lastPrice") or 0)
            volume = float(item.get("quoteVolume") or 0)
        except (TypeError, ValueError):
            continue
        if last <= 0 or change <= 0 or volume < 5_000_000:
            continue
        ranked.append((change, base))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [base for _, base in ranked[:limit]]


def _candidate_crypto() -> list[dict[str, Any]]:
    bases = list(dict.fromkeys([*_crypto_symbols(), *_binance_daily_gainers(limit=5, exclude=set(_crypto_symbols()))]))
    out: list[dict[str, Any]] = []
    for base in bases:
        try:
            horizons = _horizons_from_closes(_closes_from_binance_klines(f"{base}USDT"))
        except Exception:
            continue
        if horizons.get("close") is None:
            continue
        out.append(
            {
                "symbol": base,
                "label": base,
                "asset_class": "crypto",
                "close": horizons["close"],
                "d": horizons["d"],
                "w": horizons["w"],
                "m": horizons["m"],
            }
        )
    return out


def _candidate_yahoo(symbols: list[str], *, asset_class: str, label_fn) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not symbols:
        return out
    with ThreadPoolExecutor(max_workers=min(6, len(symbols))) as pool:
        futures = {pool.submit(_yahoo_horizons, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                horizons = future.result()
            except Exception:
                continue
            if horizons.get("close") is None:
                continue
            out.append(
                {
                    "symbol": symbol,
                    "label": label_fn(symbol),
                    "asset_class": asset_class,
                    "close": horizons["close"],
                    "d": horizons["d"],
                    "w": horizons["w"],
                    "m": horizons["m"],
                }
            )
    return out


def _metal_label(symbol: str) -> str:
    return _METAL_LABELS.get(symbol, symbol)


def _quote_label(symbol: str) -> str:
    return symbol


def _best(candidates: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    ranked = [
        row
        for row in candidates
        if isinstance(row.get(key), (int, float))
    ]
    if not ranked:
        return None
    return max(ranked, key=lambda row: float(row[key]))


def fetch_period_movers() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_crypto = pool.submit(_candidate_crypto)
        fut_metals = pool.submit(
            _candidate_yahoo,
            _metal_symbols(),
            asset_class="metals",
            label_fn=_metal_label,
        )
        fut_quotes = pool.submit(
            _candidate_yahoo,
            _quote_symbols(),
            asset_class="quotes",
            label_fn=_quote_label,
        )
        for future in (fut_crypto, fut_metals, fut_quotes):
            try:
                candidates.extend(future.result())
            except Exception:
                continue

    rows: list[dict[str, Any]] = []
    for horizon, label in (("d", "1d"), ("w", "1w"), ("m", "1m")):
        winner = _best(candidates, horizon)
        if winner is None:
            rows.append(
                {
                    "horizon": label,
                    "label": "—",
                    "asset_class": "",
                    "close": None,
                    "change_pct": None,
                }
            )
            continue
        rows.append(
            {
                "horizon": label,
                "label": str(winner.get("label") or winner.get("symbol") or "?"),
                "asset_class": str(winner.get("asset_class") or ""),
                "close": winner.get("close"),
                "change_pct": winner.get(horizon),
                "symbol": winner.get("symbol"),
            }
        )
    return rows
