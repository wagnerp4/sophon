# Plan: energy, MCP + NexusTools, compact

Parent comparison: [harness-comparison.md](harness-comparison.md). Specs: [energy.md](energy.md), [compact.md](compact.md), [../integrations/mcp/README.md](../integrations/mcp/README.md).

## Locks

1. `/energy` is not `/mode`. Permission mode stays plan/chat/agent.
2. API mode is hosted HTTP (OpenAI, Anthropic, Google, later DeepSeek). It is not the Claude Code CLI.
3. `/energy api` is a follow-up backend picker, then a swap. Local weights stay loaded. `/energy api openai` skips the provider step.
4. Daily cap default **$5**. On limit: ask and hold, then stop after `SOPHON_ENERGY_ASK_TIMEOUT_S` (60).
5. MCP inventory retraces core connectors. NexusTools win the model schema on duplicates. They wrap Local REST and extend it. Obsidian is in-process, not Cursor stdio.
6. `/compact` is not `/memory compact`. Auto-compact defaults on for **local**. API auto-compact is off unless `/compact billed on`, with an estimate and the same ask-and-hold.
7. One harness policy for every origin.

## Order

1. `/energy` regime, ledger, picker, ask-and-hold.
2. `/compact`.
3. Tool origins and `/tools` filters.
4. In-process Obsidian MCP retrace.
5. Zotero retrace and injected listing.

Out: OS sandbox, worktrees, continuable subagents, hooks, HF generate-then-display, wrapping vendor CLIs.

## Reminders

- Windows TUI is a second tree. After code: `./scripts/deploy-windows.sh --skip-venv`, then restart the **sophon** Windows Terminal profile.
- PowerShell for env examples. Double quotes for strings in Python.
- Do not wrap Claude Code or Codex.
