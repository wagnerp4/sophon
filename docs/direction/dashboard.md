# Dashboard

## Role

The dashboard is the idle and ops surface: host health, markets, news, services, weather. It is not the orchestrator. It must show whether the harness, models, and later swarms are alive without opening Chat.

## Current behavior

- Mode: `Ctrl+D` (default startup unless `--chat-first`).
- Layout from `SOPHON_DASHBOARD_TILES` (see [reference/setup.md](../reference/setup.md)).
- Tiles: host (hardware prices optional via eBay), weather (Open-Meteo), services, storage, system, markets (crypto/metals/quotes), news/papers, polymarket, twitch, optional vault.
- JSON cache: `data/dashboard/`.
- Model preload continues in the background. Progress goes to the status bar / session log, not to Rich bars behind Textual.

## Constraints

- **Read-only.** Tiles poll public or credentialed APIs. They do not start training or send mail.
- **Degrade to inert.** Missing eBay / Twitch / Google keys: tile shows inert, no crash loop.
- **No extra GUI.** If a metric cannot render in a Textual widget, it does not belong here.
- **Cache on disk.** Polling must tolerate offline. Stale timestamps should be visible.
- **GPU session.** System/host tiles should not allocate CUDA. `nvidia-ml-py` / `psutil` only.
- **Secrets.** Twitch/eBay/Google IDs stay in `.env`. Tile JSON must not persist secrets.

## Future direction

1. Occupancy line, then swarm tile: resident model VRAM, leftover, queued children, API in-flight, last placement refuse reason. Data from the subagent registry / later `one/` state, not from scraping the TUI. Spec: [subagents.md](subagents.md). Energy HUD: regime + today's USD vs daily cap ([energy.md](energy.md)).
2. Chat status already prints `Skills · Tools · MCP`. MCP is 0 until the host ships. Intended breakdown: NexusTools + MCP + Injected ([../integrations/mcp/README.md](../integrations/mcp/README.md)).
3. Services tile already probes Ollama, LM Studio, TTS, Obsidian, Zotero. Extend with harness policy summary (tools on/off).
4. Storage tile: keep volume bars. Optional `SOPHON_SYSTEM_FILETYPE_ROOTS`. Do not index `models/` on every refresh.
5. News/papers stay RSS + HN + arXiv + HF trending. Crossref live citation counts are Chat/Zotero work, not a dashboard requirement.
6. Skins tile remains optional (`SOPHON_SKINS_*` in setup notes).

## Non-goals

- Trading execution.
- Replacing Grafana.
- Auto-refresh intervals so aggressive they trip APIs (Twitch/Helix, eBay Browse).
