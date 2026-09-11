# Chat shell tools

Enable local filesystem navigation and shell execution inside orodruin chat (LM Studio tool loop). Same runner as the Workshop TUI prompt (`!` / `$` / `/shell`).

## Setup

In `.env`:

```text
ORODRUIN_SHELL_TOOLS=1
# ORODRUIN_SHELL_TIMEOUT_S=60
```

Restart chat. Check with `/tools`.

If Obsidian tools are also enabled, vault_* and shell_* are available in the same session (no mode switch).

## Tools

| Tool | Role |
|------|------|
| `shell_pwd` | Current working directory |
| `shell_cd` | Change directory (empty → home) |
| `shell_ls` | List directory |
| `shell_read` | Read text file (truncated) |
| `shell_exec` | Run a command (PowerShell on Windows) |

TUI screen switching (`Ctrl+W` Workshop) is independent. That screen is for interactive shell typing. Chat uses tool calls through the model.

The LM Studio tool loop is unlimited by default. Cap or uncap in chat with `/tool-rounds 6` or `/unlimited`. Persist with `ORODRUIN_TOOL_MAX_ROUNDS` in `.env`.
