# Obsidian vault + sophon / LM Studio

Two ways to read the same Obsidian vault:

| Client | How |
|--------|-----|
| **sophon chat** (LM Studio backend) | Phase 1: OpenAI tools → sophon calls **Local REST API** |
| **LM Studio app chat** | Phase 2: LM Studio MCP Host loads your Obsidian MCP via `mcp.json` |

## Prerequisites

1. Obsidian open with [Local REST API](https://github.com/coddingtonbear/obsidian-local-rest-api) enabled (or your existing Obsidian MCP bridge that uses it).
2. Copy the API key from the plugin settings.
3. Note the URL (often `https://127.0.0.1:27124` for HTTPS, or `http://127.0.0.1:27123` for HTTP).

## Phase 1 — sophon tool loop

Prefer the sophon repo `.env` (loaded via `load_sophon_dotenv` on startup):

```text
SOPHON_OBSIDIAN_TOOLS=1
SOPHON_OBSIDIAN_API_URL=https://127.0.0.1:27124
SOPHON_OBSIDIAN_API_KEY=paste-from-plugin
```

In PowerShell, bash-style `VAR=value` does not work. Use either `.env` above, or for the current session:

```powershell
$env:SOPHON_OBSIDIAN_TOOLS = "1"
$env:SOPHON_OBSIDIAN_API_URL = "https://127.0.0.1:27124"
$env:SOPHON_OBSIDIAN_API_KEY = "paste-from-plugin"
```

Restart sophon chat after changing `.env`. In session:

```text
/obsidian-status
can you list notes in the vault root?
search the vault for rust
```

Tools exposed to LM Studio: `vault_search`, `vault_list`, `vault_read`, `vault_recent` (read-only). sophon runs a multi-round tool loop and feeds results back to the model. The loop is unlimited by default (`/tool-rounds`, `/unlimited`, or `SOPHON_TOOL_MAX_ROUNDS`).

## Phase 2 — LM Studio native MCP

Use the **same** Obsidian MCP you already use in Cursor (`user-obsidian` / Local REST API MCP).

1. In LM Studio: **Program** tab → **Install** → **Edit mcp.json**.
2. Enable **Allow calling servers from mcp.json** in Server Settings.
3. Prefer copying the working Obsidian block from Cursor's `mcp.json` (same `mcpServers` shape). If Cursor already talks to your vault, reuse that entry unchanged when paths work from the LM Studio host process.

Generic Local REST API MCP example (only if you do not already have a Cursor entry):

```json
{
  "mcpServers": {
    "obsidian": {
      "command": "npx",
      "args": ["-y", "obsidian-mcp"],
      "env": {
        "OBSIDIAN_API_KEY": "paste-from-plugin",
        "OBSIDIAN_HOST": "127.0.0.1",
        "OBSIDIAN_PORT": "27124"
      }
    }
  }
}
```

On Windows, a `.cmd` wrapper (as in some Cursor setups) is fine if LM Studio can run that command.

Keep Obsidian running while LM Studio or sophon talk to the vault.

## Editor tree

Set `SOPHON_VAULT_PATH` to the vault filesystem root. In Editor, **View → Vault tree** (or `Ctrl+Shift+T`) retargets the project pane to that folder. The terminal cwd stays on the code workspace. **View → Project tree** switches back. `.obsidian` is hidden. Notes open in the same editor as project files.

## Notes

- sophon does **not** speak MCP over stdio for Phase 1. It uses HTTP Local REST API so the TUI stays simple.
- LM Studio app chat uses MCP Host. sophon chat uses REST tools. Same vault, two clients.
- Passing LM Studio API `integrations` from sophon is out of scope for this setup.
