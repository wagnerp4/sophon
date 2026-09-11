from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

from cli.tui.tiles.base import BaseTile, TileState, http_get_json, wrap_fetch

_WEATHER_CODES = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}


def _reverse_geocode(lat: float, lon: float) -> str:
    url = (
        "https://geocoding-api.open-meteo.com/v1/reverse"
        f"?latitude={lat}&longitude={lon}&language=en&format=json"
    )
    try:
        data = http_get_json(url)
    except Exception:
        return f"{lat:.2f},{lon:.2f}"
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        return f"{lat:.2f},{lon:.2f}"
    hit = results[0]
    name = str(hit.get("name") or "")
    admin = str(hit.get("admin1") or "")
    country = str(hit.get("country_code") or hit.get("country") or "")
    parts = [part for part in (name, admin, country) if part]
    return ", ".join(parts) if parts else f"{lat:.2f},{lon:.2f}"


def _ip_geolocate() -> tuple[float, float, str]:
    endpoints = (
        "https://ipapi.co/json/",
        "https://ipinfo.io/json",
    )
    errors: list[str] = []
    for url in endpoints:
        try:
            data = http_get_json(url, timeout_s=8.0)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is None or lon is None:
            loc = str(data.get("loc") or "")
            if "," in loc:
                lat_s, lon_s = loc.split(",", 1)
                lat, lon = float(lat_s), float(lon_s)
            else:
                continue
        lat_f, lon_f = float(lat), float(lon)
        city = str(data.get("city") or "")
        region = str(data.get("region") or data.get("region_code") or "")
        country = str(data.get("country_code") or data.get("country") or "")
        label_parts = [part for part in (city, region, country) if part]
        label = ", ".join(label_parts) if label_parts else _reverse_geocode(lat_f, lon_f)
        return lat_f, lon_f, label
    raise ValueError(
        "could not auto-detect location "
        f"({'; '.join(errors) if errors else 'no geo endpoint'}); "
        "set ORODRUIN_WEATHER_LAT/LON or ORODRUIN_WEATHER_QUERY"
    )


def _resolve_coords() -> tuple[float, float, str]:
    lat_raw = os.environ.get("ORODRUIN_WEATHER_LAT", "").strip()
    lon_raw = os.environ.get("ORODRUIN_WEATHER_LON", "").strip()
    if lat_raw and lon_raw:
        label = os.environ.get("ORODRUIN_WEATHER_QUERY", "").strip()
        lat_f, lon_f = float(lat_raw), float(lon_raw)
        if not label:
            label = _reverse_geocode(lat_f, lon_f)
        return lat_f, lon_f, label
    query = os.environ.get("ORODRUIN_WEATHER_QUERY", "").strip()
    if query:
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?name={quote(query)}&count=1&language=en&format=json"
        )
        data = http_get_json(geo_url)
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            raise ValueError(f"no geocode result for {query!r}")
        hit = results[0]
        name = str(hit.get("name") or query)
        admin = str(hit.get("admin1") or "")
        country = str(hit.get("country_code") or hit.get("country") or "")
        label = ", ".join(part for part in (name, admin, country) if part)
        return float(hit["latitude"]), float(hit["longitude"]), label
    return _ip_geolocate()


def _want_aqi() -> bool:
    raw = os.environ.get("ORODRUIN_WEATHER_AQI", "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _hourly_strip(hourly: dict[str, Any] | None, *, count: int = 4) -> str:
    if not isinstance(hourly, dict):
        return ""
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    codes = hourly.get("weather_code") or []
    if not isinstance(times, list) or not isinstance(temps, list):
        return ""
    parts: list[str] = []
    for idx in range(min(count, len(times), len(temps))):
        stamp = str(times[idx])
        hour = stamp[11:16] if len(stamp) >= 16 else stamp
        code = int(codes[idx]) if idx < len(codes) and codes[idx] is not None else 0
        short = _WEATHER_CODES.get(code, "?")
        if " " in short:
            short = short.split(" ", 1)[0]
        parts.append(f"{hour} {temps[idx]}C {short}")
    return "  ".join(parts)


def _fetch_aqi(lat: float, lon: float, tz: str) -> dict[str, Any] | None:
    if not _want_aqi():
        return None
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&current=european_aqi,pm2_5,pm10,us_aqi"
        f"&timezone={quote(tz)}"
    )
    try:
        data = http_get_json(url, timeout_s=8.0)
    except Exception:
        return None
    current = data.get("current") if isinstance(data, dict) else None
    return current if isinstance(current, dict) else None


def _fetch_weather() -> TileState:
    lat, lon, label = _resolve_coords()
    tz = os.environ.get("ORODRUIN_WEATHER_TZ", "").strip() or "auto"
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
        "weather_code,wind_speed_10m,precipitation"
        "&hourly=temperature_2m,weather_code,precipitation_probability"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
        "&forecast_days=3"
        "&forecast_hours=12"
        f"&timezone={quote(tz)}"
    )
    data = http_get_json(url)
    current = data.get("current") if isinstance(data, dict) else None
    daily = data.get("daily") if isinstance(data, dict) else None
    hourly = data.get("hourly") if isinstance(data, dict) else None
    if not isinstance(current, dict):
        raise ValueError("open-meteo response missing current")
    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    precip = current.get("precipitation")
    code = int(current.get("weather_code") or 0)
    wind = current.get("wind_speed_10m")
    desc = _WEATHER_CODES.get(code, f"code {code}")
    lines = [
        "weather",
        label,
        f"{temp} C (feels {feels} C)  {desc}",
        f"wind {wind} km/h  humidity {humidity}%  precip {precip} mm",
    ]
    strip = _hourly_strip(hourly if isinstance(hourly, dict) else None, count=4)
    if strip:
        lines.append(f"next  {strip}")
    if isinstance(daily, dict):
        tmax = daily.get("temperature_2m_max") or []
        tmin = daily.get("temperature_2m_min") or []
        day_precip = daily.get("precipitation_sum") or []
        day_codes = daily.get("weather_code") or []
        if len(tmax) >= 1 and len(tmin) >= 1:
            lines.append(f"today {tmin[0]}-{tmax[0]} C  rain {day_precip[0] if day_precip else '?'} mm")
        if len(tmax) >= 2 and len(tmin) >= 2:
            tomorrow_code = int(day_codes[1]) if len(day_codes) > 1 and day_codes[1] is not None else 0
            tomorrow_desc = _WEATHER_CODES.get(tomorrow_code, "?")
            lines.append(
                f"tomorrow {tmin[1]}-{tmax[1]} C  rain "
                f"{day_precip[1] if len(day_precip) > 1 else '?'} mm  {tomorrow_desc}"
            )
    aqi = _fetch_aqi(lat, lon, tz)
    if aqi:
        eu = aqi.get("european_aqi")
        pm25 = aqi.get("pm2_5")
        lines.append(f"aqi EU {eu}  PM2.5 {pm25}")
    return TileState(
        ok=True,
        text="\n".join(lines),
        fetched_at=time.time(),
        payload={
            "lat": lat,
            "lon": lon,
            "label": label,
            "current": current,
            "daily": daily,
            "hourly": hourly,
            "aqi": aqi,
            "source": "open-meteo",
        },
    )


class WeatherTile(BaseTile):
    tile_id = "weather"
    refresh_s = 900.0

    def fetch(self, app: Any) -> TileState:
        return wrap_fetch(self.tile_id, "weather", _fetch_weather)
