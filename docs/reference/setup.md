### Local shell tools in chat

The TUI **Workshop** screen already runs a local shell. The same capability is available inside Chat (alongside vault tools when both are enabled):

```text
ORODRUIN_SHELL_TOOLS=1
# ORODRUIN_SHELL_TIMEOUT_S=60
```

With `ORODRUIN_OBSIDIAN_TOOLS=1` and `ORODRUIN_SHELL_TOOLS=1`, LM Studio chat gets speak + vault_* + shell_* together. No mode switch. Check with `/tools`.

On Windows, `shell_exec` uses PowerShell. TUI screens (**Dashboard** / **Workshop** / **Editor** / **Chat** via `Ctrl+D` / `Ctrl+W` / `Ctrl+E` / `Ctrl+G`) are UI layouts. Editor uses the workshop project tree for `.py` / `.md` / `.svg` / `.stl` / tables / `.ipynb` / `.pdf` (Python full-width with syntax highlighting when `textual[syntax]` is installed; Markdown, SVG, STL, CSV/TSV/JSONL/XLSX, notebooks, and PDFs use a preview pane). The editor Terminal tab, Workshop shell (`!` / `$`), and chat shell tools share the same runner.

Optional dashboard env (loaded from `.env` in the repo root via `load_orodruin_dotenv`):

```bash
# layout: host|weather, services|storage, system (full), markets (full), news (full), polymarket, twitch (full)
ORODRUIN_DASHBOARD_TILES=host,weather,services,storage,system,markets,news,polymarket,twitch
# alias: gpu -> system. hardware is embedded in host (standalone hardware tile skipped if host is present)

# weather (Open-Meteo, no key). Default: auto IP geolocation.
# ORODRUIN_WEATHER_QUERY=Munich
# ORODRUIN_WEATHER_LAT=48.137
# ORODRUIN_WEATHER_LON=11.576
# ORODRUIN_WEATHER_TZ=Europe/Berlin
# ORODRUIN_WEATHER_AQI=1

# markets tile sections (already live)
ORODRUIN_CRYPTO_SYMBOLS=BTC,ETH,SOL
# ORODRUIN_CRYPTO_PROVIDER=kraken
ORODRUIN_METALS_SYMBOLS=GC=F,SI=F
# ORODRUIN_METALS_SYMBOLS=GC=F,SI=F,PL=F,HG=F
ORODRUIN_QUOTES_SYMBOLS=AAPL,MSFT,NVDA,GOOGL,AMZN,META,SPY,QQQ,IWM
# ORODRUIN_QUOTES_PROVIDER=yahoo

# hardware section inside host (eBay Browse; inert until credentials are set)
# ORODRUIN_HARDWARE_QUERIES=RTX 5090,RTX 4090,RTX 3090 Ti,A100,Ryzen 9 7950X,DDR5 64GB
# ORODRUIN_EBAY_CLIENT_ID=...
# ORODRUIN_EBAY_CLIENT_SECRET=...

# news | papers (RSS/HN left; arXiv picks + HF daily trending right)
# ORODRUIN_NEWS_FEEDS=https://www.tagesschau.de/xml/rss2/,https://www.br.de/nachrichten/rss
# ORODRUIN_NEWS_HN=1
# ORODRUIN_NEWS_LIMIT=6
# ORODRUIN_PAPERS_LIMIT=4
# ORODRUIN_PAPERS_HF=1
# ORODRUIN_NEWS_ARXIV_QUERY=all:"sound event detection" OR all:SELD||all:"brain-computer interface" OR cat:q-bio.NC
# ORODRUIN_NEWS_ARXIV_CAT=

# polymarket (Gamma API, keyless). Empty = top by 24h volume.
# ORODRUIN_POLYMARKET_MARKETS=
# ORODRUIN_POLYMARKET_LIMIT=5

# twitch (Helix app credentials required). Profile is a label for this env set.
# ORODRUIN_TWITCH_CLIENT_ID=...
# ORODRUIN_TWITCH_CLIENT_SECRET=...
# ORODRUIN_TWITCH_PROFILE=home
# ORODRUIN_TWITCH_MODE=top
# ORODRUIN_TWITCH_LIMIT=8
# ORODRUIN_TWITCH_LANGUAGE=en
# ORODRUIN_TWITCH_GAME=
# ORODRUIN_TWITCH_CHANNELS=
```

#### What you can track today by editing `.env` only

| Goal | Env | Notes |
| --- | --- | --- |
| BTC / ETH / SOL (+ more) | `ORODRUIN_CRYPTO_SYMBOLS` | Kraken pairs; add e.g. `XRP,DOGE` if the pair maps cleanly |
| GOLD / SILVER / more | `ORODRUIN_METALS_SYMBOLS` | Yahoo futures: `GC=F` gold, `SI=F` silver, `PL=F` platinum, `HG=F` copper, `PA=F` palladium |
| Big tech | `ORODRUIN_QUOTES_SYMBOLS` | Yahoo tickers: `AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,...` |
| ETF / indices | same `ORODRUIN_QUOTES_SYMBOLS` | e.g. `SPY,QQQ,IWM,DIA,VGK,EEM` or `^GDAXI` style where Yahoo accepts it |
| Cross-asset movers | (auto in markets) | Best 1d / 1w / 1m gainer across crypto + metals + quotes |
| Local weather + AQI | `ORODRUIN_WEATHER_*` | Prefer `Munich` / coords; hourly + tomorrow + EU AQI |
| Used GPU/CPU asking prices | `ORODRUIN_HARDWARE_*` + eBay keys | Shown inside the **host** tile |
| Local services | (auto) | **services** tile probes Ollama, LM Studio, TTS, Obsidian |
| Vault glance | `ORODRUIN_OBSIDIAN_*` | optional **vault** tile (add `vault` to `ORODRUIN_DASHBOARD_TILES`): ping + daily note + root listing |
| Storage | psutil + optional `ORODRUIN_SYSTEM_FILETYPE_ROOTS` | **storage** tile: volume fill bars + sampled filetype sizes |
| News / papers | `ORODRUIN_NEWS_*`, `ORODRUIN_PAPERS_*` | Left: RSS + HN. Right: arXiv topic picks + HF daily trending |
| Prediction markets | `ORODRUIN_POLYMARKET_*` | Gamma API; optional event slugs |
| Live Twitch | `ORODRUIN_TWITCH_*` | App client id/secret; `MODE=top` or `channels` |

Restart the TUI after `.env` changes (`.\scripts\start-orodruin-chat.ps1`). No need to reinstall autostart.

#### Remaining ideas (need a new tile later; suggested env names)

```bash
# CS skins (Skinport public API)
# ORODRUIN_SKINS_QUERY=Karambit|Doppler,Butterfly Knife|Fade
# ORODRUIN_SKINS_CURRENCY=EUR

# Optional later tiles still use the same layout switch:
# ORODRUIN_DASHBOARD_TILES=...,skins
```

| Source | Key? | Feasibility |
| --- | --- | --- |
| Skinport | no | High; Steam priceoverview is too fragile for polling |
| Twitch user follows | user OAuth (`user:read:follows`) | Medium; app token only covers top/channel login lists today |
| TradingView | n/a | Skip; no public data API (Yahoo/Kraken already cover prices) |
| Geizhals / Idealo live new-part prices | OAuth / scrape | Medium; prefer eBay Browse for used asking prices |

#### Weather APIs (what we use vs options)

| API | Key | Role |
| --- | --- | --- |
| **Open-Meteo** (current) | no | Current + hourly + daily forecast |
| Open-Meteo Geocoding | no | City name → lat/lon (`ORODRUIN_WEATHER_QUERY`) |
| Open-Meteo Air Quality | no | EU AQI / PM2.5 (`ORODRUIN_WEATHER_AQI=1`) |
| ipapi.co / ipinfo.io | no | Auto “local” when no weather env is set |
| Bright Sky (DWD) | no | Germany-specific alternative if you want DWD stations |

Tile JSON cache lives under `data/dashboard/`. Default layout is **host | weather**, **services | storage**, then **system**, **markets**, **news**, **polymarket**, **twitch**. Hardware prices render inside host. System telemetry needs `nvidia-ml-py` and `psutil` (pulled by `--extra tui`) and works best on the Windows venv. Hardware stays inert until eBay Browse credentials are set. Twitch stays inert until Helix app credentials are set.

### Autostart on Windows login

One-time install (PowerShell):

```powershell
cd C:\Software\Python\NLP\Personal\orodruin
.\scripts\install-orodruin-autostart.ps1
# optional preset:
.\scripts\install-orodruin-autostart.ps1 -Preset llama2_7b_chat
```

That drops a Startup shortcut which opens Windows Terminal and runs `.venv\Scripts\orodruin-cli.exe chat` (no `uv` each time). Manual launch:

```powershell
.\scripts\start-orodruin-chat.ps1
```

Remove with `.\scripts\uninstall-orodruin-autostart.ps1`.
