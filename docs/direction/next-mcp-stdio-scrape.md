# Plan: MCP stdio + one scrape server

Umbrella: [next-mcp-retrieval-loop.md](next-mcp-retrieval-loop.md). Spec baseline: [../integrations/mcp/README.md](../integrations/mcp/README.md). Search today: [../integrations/search/README.md](../integrations/search/README.md).

## Goal

Ship a real **stdio** MCP client for servers sophon does not wrap in-process, then allowlist **one** scrape/fetch server so the model can pull page text as evidence. Search stays NexusTools.

## Why scrape, not Brave-as-primary

`/search` and `web_search` already cover SERP and verticals (SearXNG, CSE, arXiv, …). The hole is **read the URL**: HTML body / cleaned markdown for a known link. Firecrawl-class MCP (or an equivalent fetch/scrape stdio server) fills that. Brave Search MCP is optional later as a second allowlisted row, not a replacement for `/search`.

## Locks

1. Transport: stdio first. HTTP MCP later if needed. Obsidian/Zotero stay in-process retrace.
2. Registry merges origins. NexusTools win name collisions. MCP scrape tools get distinct names (`mcp:firecrawl/*` or server-prefixed).
3. Same harness table: deny / ask / allow. Plan mode denies write-shaped MCP tools. Scrape is read by default. `ask` if the tool can hit arbitrary URLs without a host allowlist.
4. No `/mcp call` in v1. Human debug: `/mcp status`, `/mcp list`, `/tools mcp`.
5. `SOPHON_MCP=0` disables the origin. Yaml has no secrets.
6. First test bed: `/energy api` or LM Studio tool loop. HF generate-then-display waits on backend work (unchanged from mcp README).

## Substeps

1. **Host client**
   - Python MCP SDK stdio session in `src/integrations/mcp/host.py` (extend beyond inventory/retrace).
   - Spawn via PowerShell-safe argv list. No `cmd.exe` string concat.
   - Lifecycle: connect on chat start (or first `/tools mcp`), disconnect on exit. Crash of child → MCP count 0 + error in `/mcp status`, not a TUI hang.

2. **Allowlist yaml**
   - `.sophon/mcp.yaml` (shareable) + `.sophon/mcp.local.yaml` (gitignored).
   - Sketch:

```yaml
servers:
  - id: firecrawl
    transport: stdio
    command: npx
    args: ["-y", "firecrawl-mcp"]
    env_from: ["FIRECRAWL_API_KEY"]
```

   - Exact package name and args confirmed at implement time against the upstream README. Substitute another scrape MCP if Firecrawl is a bad fit. Keep **one** server in the first merge.

3. **Schema + policy**
   - List tools into the registry with `origin=mcp`, `connector_id=<id>`.
   - Trace: origin, server id, tool name, duration, permission decision.
   - HUD: `MCP: N` reflects connected tools (not yaml rows alone).

4. **Pilot exercise**
   - Agent mode: “scrape this URL and summarize”. Evidence lands in the turn trace.
   - Refuse to treat scrape output as trusted code. No auto `shell_exec` of fetched scripts.

5. **Docs / env**
   - Document `FIRECRAWL_API_KEY` (or chosen server key) in `.env.example` if present.
   - Point search README “Out of this slice” at this plan for scrape. Brave remains a later optional server row.

## Done when

- `/tools mcp` lists tools from the one stdio server while it is up.
- Model can call at least one scrape/fetch tool under harness policy.
- `/search` behavior unchanged.
- Windows deploy tree receives the same code. Restart **sophon** profile required.

## Out

- Auto-install / Hub marketplace
- Multiple scrape providers in v1
- Replacing native `web_search`
- Scrapling as a first-party NexusTool (can revisit if MCP path fails)
- e2b / browser MCP as the first server
