# Plan: MCP stdio, retrieval ignore/tags, solve-until-done

Parent map: [README.md](README.md). Prior iteration: [next-energy-mcp-compact.md](next-energy-mcp-compact.md).

These three are the **next direction** after energy / in-process MCP retrace / compact. They come from the vault note `Personal KB/Projects/Sophon.md`, cut down to what the local harness can own without cloud-GPU fantasy.

| # | Plan | Spec / detail |
| --- | --- | --- |
| 1 | MCP stdio + one search/scrape server | [next-mcp-stdio-scrape.md](next-mcp-stdio-scrape.md) |
| 2 | Retrieval `.ignore` + tagging | [next-retrieval-ignore-tags.md](next-retrieval-ignore-tags.md) |
| 3 | Solve-until-done loop (local + API only) | [next-solve-until-done.md](next-solve-until-done.md) |

## Locks

1. Do not replace NexusTools `/search` or `web_search` with an MCP search server. Native search stays. Stdio MCP fills a gap (page scrape / fetch) or adds an optional second origin behind the same harness policy.
2. Do not auto-install MCP from a marketplace. One allowlisted server in `.sophon/mcp.yaml`. Credentials stay in `.env`.
3. Retrieval ignore and tags are **corpus policy**, not a second RAG product. LEANN + LightRAG + Adaptive-RAG stay. Rebuild remains `/rag-index` / `sophon-rag-index`.
4. Solve-until-done uses the **existing** energy regimes (`/energy local|api`) and the existing subagent depth caps. No remote GPU pods, no free-tier Kaggle/Colab marketplace, no bounty cash loop.
5. Maker/checker is evidence against artifacts (files, tests, STATE.md), not model self-grade.
6. tmux, agent VMs / e2b fleets, and harnessrouter-style external CLI routing stay out of this iteration.

## Order

1. **MCP stdio + one scrape server** — closes the transport hole left after in-process Obsidian/Zotero retrace. Unlocks external tools without inventing another I/O channel.
2. **Retrieval `.ignore` + tagging** — improves what the model sees before longer agent loops burn tokens on junk.
3. **Solve-until-done** — needs reliable tools and a cleaner corpus. Depends on (1) for scrape evidence and on (2) for gated context. Subagent spawn seam already exists ([subagents.md](subagents.md)).

Do not start (3) as a swarm overlay. Start as one-session outer loop + `STATE.md`. Orchestrator fan-out stays [orchestrator.md](orchestrator.md).

## Out

- Wrapping Claude Code / Codex as MCP children
- Continuous self-rewrite of the harness
- Paid cloud GPU bounty swarms
- Free-tier backend listing UI
- Vision chat product path
- Jobs queue / Seedtable (already shipped; separate track)

## Reminders

- Windows TUI is a second tree. After code: `./scripts/deploy-windows.sh --skip-venv`, then restart the **sophon** Windows Terminal profile.
- PowerShell for env examples. Double quotes for strings in Python.
- Update [TODO.md](../TODO.md) item status when a substep lands. Do not treat this folder as the backlog of record.
