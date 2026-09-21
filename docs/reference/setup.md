### Local shell tools in chat

Local shell execution is available inside Chat (alongside vault tools when both are enabled):

```text
SOPHON_SHELL_TOOLS=1
# SOPHON_SHELL_TIMEOUT_S=60
```

With `SOPHON_OBSIDIAN_TOOLS=1` and `SOPHON_SHELL_TOOLS=1`, LM Studio chat gets speak + vault_* + shell_* together. `shell_exec` still needs harness approval (1 this time / 2 allow-list / 3 decline) unless the prefix is in `.sophon/harness.local.yaml`. Check with `/tools` and `/permissions`.

Editor chat (LM Studio) can propose file edits without writing disk until you accept them. Tools are on by default:

```text
# SOPHON_EDITOR_TOOLS=0
```

Memory layer is on by default. Disable with `SOPHON_MEMORY_DB=0`.

```text
SOPHON_MEMORY_DB=data/memory/memory.db
SOPHON_MEMORY_TOOLS=1
```

Skills are on by default. Disable with `SOPHON_SKILLS=0`.

```text
SOPHON_SKILL_TOOLS=1
```

`/skill list` shows discovered `SKILL.md` folders. `/skill create <name>` queues a Review write under `.sophon/skills`.

`editor_read` / `editor_propose_edit` / `editor_status` queue a combined changeset. In the editor, open the **Review** tab (View → Code Review) to see affected vs edited files, then **Accept all** / **Decline all**. **Undo batch** / **Redo batch** revert or re-apply the last accepted set. Slash commands: `/assist status|accept|decline|undo|redo`.

On Windows, `shell_exec` uses PowerShell. TUI screens (**Dashboard** / **Editor** / **Chat** via `Ctrl+D` / `Ctrl+E` / `Ctrl+G`) are UI layouts. Editor uses the project tree for `.py` / `.md` / `.svg` / `.stl` / tables / `.ipynb` / `.pdf` / images (Python full-width with syntax highlighting when `textual[syntax]` is installed; Markdown, SVG, STL, CSV/TSV/JSONL/XLSX, notebooks, PDFs, and PNG/JPG/WebP use a preview pane). PDF preview defaults to page text. Raster is the current page only, via Sixel/Kitty when the host terminal supports it. The editor Terminal tab and chat shell tools share the same runner.

Optional dashboard env (loaded from `.env` in the repo root via `load_sophon_dotenv`):

```bash
# layout: host|weather, services|storage, system (full), markets (full), news (full), polymarket, twitch (full)
SOPHON_DASHBOARD_TILES=host,weather,services,storage,system,markets,news,polymarket,twitch
# alias: gpu -> system. hardware is embedded in host (standalone hardware tile skipped if host is present)

# weather (Open-Meteo, no key). Default: auto IP geolocation.
# SOPHON_WEATHER_QUERY=Munich
# SOPHON_WEATHER_LAT=48.137
# SOPHON_WEATHER_LON=11.576
# SOPHON_WEATHER_TZ=Europe/Berlin
# SOPHON_WEATHER_AQI=1

# markets tile sections (already live)
SOPHON_CRYPTO_SYMBOLS=BTC,ETH,SOL
# SOPHON_CRYPTO_PROVIDER=kraken
SOPHON_METALS_SYMBOLS=GC=F,SI=F
# SOPHON_METALS_SYMBOLS=GC=F,SI=F,PL=F,HG=F
SOPHON_QUOTES_SYMBOLS=AAPL,MSFT,NVDA,GOOGL,AMZN,META,SPY,QQQ,IWM
# SOPHON_QUOTES_PROVIDER=yahoo

# hardware section inside host (eBay Browse; inert until credentials are set)
# SOPHON_HARDWARE_QUERIES=RTX 5090,RTX 4090,RTX 3090 Ti,A100,Ryzen 9 7950X,DDR5 64GB
# SOPHON_EBAY_CLIENT_ID=...
# SOPHON_EBAY_CLIENT_SECRET=...

# news | papers (RSS/HN left; arXiv picks + HF daily trending right)
# SOPHON_NEWS_FEEDS=https://www.tagesschau.de/xml/rss2/,https://www.br.de/nachrichten/rss
# SOPHON_NEWS_HN=1
# SOPHON_NEWS_LIMIT=6
# SOPHON_PAPERS_LIMIT=4
# SOPHON_PAPERS_HF=1
# SOPHON_NEWS_ARXIV_QUERY=all:"sound event detection" OR all:SELD||all:"brain-computer interface" OR cat:q-bio.NC
# SOPHON_NEWS_ARXIV_CAT=

# polymarket (Gamma API, keyless). Empty = top by 24h volume.
# SOPHON_POLYMARKET_MARKETS=
# SOPHON_POLYMARKET_LIMIT=5

# twitch (Helix app credentials required). Profile is a label for this env set.
# SOPHON_TWITCH_CLIENT_ID=...
# SOPHON_TWITCH_CLIENT_SECRET=...
# SOPHON_TWITCH_PROFILE=home
# SOPHON_TWITCH_MODE=top
# SOPHON_TWITCH_LIMIT=8
# SOPHON_TWITCH_LANGUAGE=en
# SOPHON_TWITCH_GAME=
# SOPHON_TWITCH_CHANNELS=

# Google (Gmail/Drive OAuth + Chrome bookmark export + Custom Search)
# SOPHON_GOOGLE_CLIENT_SECRETS=
# SOPHON_GOOGLE_TOOLS=1
# SOPHON_BOOKMARKS_PATH=C:\Notes\Obsidian Notes\KB\Data\Private Data\Chrome\Bookmark Exports
# SOPHON_GOOGLE_CSE_KEY=
# SOPHON_GOOGLE_CSE_CX=
# SOPHON_WEB_SEARCH_TOOLS=1

# Overleaf Premium Git (editor tree + LM Studio list/read tools)
# SOPHON_OVERLEAF_TOOLS=1
# SOPHON_OVERLEAF_GIT_TOKEN=
# SOPHON_OVERLEAF_PROJECT_ID=
```

#### What you can track today by editing `.env` only

| Goal | Env | Notes |
| --- | --- | --- |
| BTC / ETH / SOL (+ more) | `SOPHON_CRYPTO_SYMBOLS` | Kraken pairs; add e.g. `XRP,DOGE` if the pair maps cleanly |
| GOLD / SILVER / more | `SOPHON_METALS_SYMBOLS` | Yahoo futures: `GC=F` gold, `SI=F` silver, `PL=F` platinum, `HG=F` copper, `PA=F` palladium |
| Big tech | `SOPHON_QUOTES_SYMBOLS` | Yahoo tickers: `AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,...` |
| ETF / indices | same `SOPHON_QUOTES_SYMBOLS` | e.g. `SPY,QQQ,IWM,DIA,VGK,EEM` or `^GDAXI` style where Yahoo accepts it |
| Cross-asset movers | (auto in markets) | Best 1d / 1w / 1m gainer across crypto + metals + quotes |
| Local weather + AQI | `SOPHON_WEATHER_*` | Prefer `Munich` / coords; hourly + tomorrow + EU AQI |
| Used GPU/CPU asking prices | `SOPHON_HARDWARE_*` + eBay keys | Shown inside the **host** tile |
| Local services | (auto) | **services** tile probes Ollama, LM Studio, TTS, Obsidian, Zotero |
| Vault glance | `SOPHON_OBSIDIAN_*` | optional **vault** tile (add `vault` to `SOPHON_DASHBOARD_TILES`): ping + daily note + root listing |
| Storage | psutil + optional `SOPHON_SYSTEM_FILETYPE_ROOTS` | **storage** tile: volume fill bars + sampled filetype sizes |
| News / papers | `SOPHON_NEWS_*`, `SOPHON_PAPERS_*` | Left: RSS + HN. Right: arXiv topic picks + HF daily trending |
| Prediction markets | `SOPHON_POLYMARKET_*` | Gamma API; optional event slugs |
| Live Twitch | `SOPHON_TWITCH_*` | App client id/secret; `MODE=top` or `channels` |
| Google mail / Drive / bookmarks / web search | `SOPHON_GOOGLE_*`, `SOPHON_BOOKMARKS_PATH`, `SOPHON_GOOGLE_CSE_*` | See [docs/integrations/google/README.md](../integrations/google/README.md) |
| Overleaf Git list / read | `SOPHON_OVERLEAF_*` | See [docs/integrations/overleaf/README.md](../integrations/overleaf/README.md) |

Restart the TUI after `.env` changes. From WSL, copy `.env` with `sophon-cli deploy-windows --skip-venv`, then `.\scripts\start-sophon-chat.ps1` on Windows. No need to reinstall autostart.

#### Remaining ideas (need a new tile later; suggested env names)

```bash
# CS skins (Skinport public API)
# SOPHON_SKINS_QUERY=Karambit|Doppler,Butterfly Knife|Fade
# SOPHON_SKINS_CURRENCY=EUR

# Optional later tiles still use the same layout switch:
# SOPHON_DASHBOARD_TILES=...,skins
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
| Open-Meteo Geocoding | no | City name → lat/lon (`SOPHON_WEATHER_QUERY`) |
| Open-Meteo Air Quality | no | EU AQI / PM2.5 (`SOPHON_WEATHER_AQI=1`) |
| ipapi.co / ipinfo.io | no | Auto “local” when no weather env is set |
| Bright Sky (DWD) | no | Germany-specific alternative if you want DWD stations |

Tile JSON cache lives under `data/dashboard/`. Default layout is **host | weather**, **services | storage**, then **system**, **markets**, **news**, **polymarket**, **twitch**. Hardware prices render inside host. System telemetry needs `nvidia-ml-py` and `psutil` (pulled by `--extra tui`) and works best on the Windows venv. Hardware stays inert until eBay Browse credentials are set. Twitch stays inert until Helix app credentials are set.

### Autostart on Windows login

The Windows tree is a deploy copy (`C:\Software\Python\NLP\Personal\sophon`). Edit in WSL, then:

```bash
sophon-cli deploy-windows
```

One-time autostart (Store PowerShell):

```powershell
cd C:\Software\Python\NLP\Personal\sophon
.\scripts\install-sophon-autostart.ps1
.\scripts\install-sophon-autostart.ps1 -Preset llama2_7b_chat
```

That drops a Startup shortcut which runs Store `pwsh.exe` and `scripts\start-sophon-chat.ps1`. Manual launch: the Windows Terminal **sophon** profile, or:

```powershell
.\scripts\start-sophon-chat.ps1
```

Remove with `.\scripts\uninstall-sophon-autostart.ps1`.
