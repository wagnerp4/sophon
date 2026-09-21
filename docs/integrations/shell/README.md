# Chat shell tools

Enable local filesystem navigation and shell execution inside sophon chat (LM Studio tool loop). Same runner as the Editor Terminal tab.

## Setup

In `.env`:

```text
SOPHON_SHELL_TOOLS=1
# SOPHON_SHELL_TIMEOUT_S=60
```

Restart chat. Check with `/tools`. `/permissions` lists allow/ask/deny. `/mode agent` is required for `shell_exec` (plan and chat deny it).

If Obsidian tools are also enabled, vault_* and shell_* are available in the same session. Harness mode is separate from tool packs.

## Tools

| Tool | Role |
|------|------|
| `shell_pwd` | Current working directory |
| `shell_cd` | Change directory (empty → home) |
| `shell_ls` | List directory |
| `shell_read` | Read text file (truncated) |
| `shell_exec` | Run a command (PowerShell on Windows) |

Interactive shell typing lives in the Editor Terminal tab. Chat uses tool calls through the model.

The LM Studio tool loop is unlimited by default. Cap or uncap in chat with `/tool-rounds 6` or `/unlimited`. Persist with `SOPHON_TOOL_MAX_ROUNDS` in `.env`.
