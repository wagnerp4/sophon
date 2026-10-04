# MCP and NexusTools

MCP-first for **model-facing I/O**. NexusTools for the **slash-command family** and for HITL that MCP cannot own (Review, 1/2/3 shell, spawn). One permission table. One owner per capability.

This is an allowlist, not plugin soup. A server is listed in `.sophon/mcp.yaml` (shareable) plus `.sophon/mcp.local.yaml` (gitignored). The model does not install servers. Wrapping Claude Code or Codex as an MCP child is still a non-goal.

## Confirm the split, with corrections

The requested shape is sound:

- Human chat functional commands stay NexusTools (`/search`, `/memory`, `/skill`, `/obsidian-status`, …). `/help` lists that family only.
- MCP tools are for the model’s background tool loop. They are not slash verbs.
- `/tools` is the inventory. Stats can show both counts and a total.

Three corrections so this does not become a second ungated channel.

1. **One harness policy.** MCP `tools/call` goes through `Tool(specifier)` deny/ask/allow the same way `shell_exec` does. Env flags still hide a pack from the schema. They still do not authorize a call. See [../direction/harness.md](../direction/harness.md).

2. **NexusTools win duplicates.** Obsidian, Zotero, Overleaf, Google, and `web_search` are core NexusTools. The MCP inventory retraces the same calls (`origin=mcp`, `shadowed_by=nexus`). The model schema keeps the NexusTools name. NexusTools wrap Local REST (or a later MCP client) and then extend it with slash commands, policy, and the vault index. Do not drop `vault_*` when the retrace is live.

3. **HITL stays native.** These never become “MCP-only”:
   - `shell_exec` / 1/2/3 / circuit breakers
   - `editor_propose_edit` / Review
   - `memory_propose`
   - `subagent` / `subagent_fork`
   The model may still call them as NexusTools in the schema. MCP must not accept a patch or persist an allow-list grant.

Injected tools (LM Studio `mcp.json` or provider-side tools that appear on the wire without a sophon registration) are a **third origin**. List them. Gate them. Do not put them in `/help`.

## Three origins

| Origin | Who calls it | Human slash | Model schema | Examples |
| --- | --- | --- | --- | --- |
| `nexus` | Chat `/commands` and the model schema | yes | yes (preferred on duplicates) | `web_search`, `vault_*`, `zotero_*`, `shell_exec`, `editor_propose_edit`, `subagent` |
| `mcp` | Model tool loop only | no (`/mcp status` is debug) | yes | Obsidian MCP, Zotero MCP, filesystem MCP, GitHub MCP |
| `injected` | Model tool loop, if the backend forwards them | no | yes, after gate | LM Studio host tools not in our yaml |

```text
surfaces(nexus)    = slash + model (preferred)
surfaces(mcp)      = inventory retrace when a NexusTool owns the name, else model
surfaces(injected) = model
```

HUD / status already has a slot:

```text
Skills: 12 · Tools: 18 · MCP: 0
```

Intended:

```text
Skills: 12 · NexusTools: 18 · MCP: 24 · Injected: 3
```

Short form when width is tight: `Tools: 45 (18n+24m+3i)`. `/tools` always prints the breakdown. Do not hide NexusTools inside the MCP count.

## `/tools` and `/help`

`/help` stays grouped slash sections (session, model, generation, tools, speech, memory, skills, eval, train, later energy). The `tools` **section** documents NexusTools commands (`/search`, `/permissions`, `/obsidian-status`, …). It does not dump MCP names.

`/tools` prints inventory:

```text
/tools                 totals + enabled packs + origins
/tools nexus           NexusTools only (slash + model)
/tools mcp             connected servers, tool names, policy
/tools injected        backend-forwarded names
/tools all             everything
```

Debug only, not a daily verb:

```text
/mcp status
/mcp list
```

No `/mcp call SERVER TOOL {json}` in v1. That would duplicate the model loop and skip traces. If a human needs to exercise a server, they ask the model in agent mode or we add that verb later behind plan-mode deny.

## Registry

One in-process registry. MCP is a transport. NexusTools is a transport. The harness sees `ToolSpec(name, origin, surfaces, connector_id)`.

Suggested layout:

- `src/harness/tools/registry.py` - merge origins, apply deny/ask/allow, expose schema for the backend.
- `src/integrations/mcp/host.py` - stdio/HTTP MCP client, allowlist from yaml.
- Existing packs stay under `src/integrations/{obsidian,zotero,google,overleaf,search,shell,...}` and **register** instead of being imported ad hoc from `chat.py`.

`.sophon/mcp.yaml` sketch:

```yaml
servers:
  - id: obsidian
    transport: inprocess
  - id: zotero
    transport: inprocess
```

Obsidian on Windows does **not** reuse Cursor `mcp.json` stdio. That would require Node and a second implementation that loses to NexusTools. The in-process provider delegates to the Local REST client. Stdio is only for servers sophon does not wrap. PowerShell quoting if a stdio row is added later. Do not shell out through `cmd.exe`.

`SOPHON_MCP=0` disables the origin (HUD MCP stays 0). `SOPHON_MCP_DISABLED=obsidian,zotero` hides servers.

Credentials stay in `.env`. Yaml must not contain API keys.

## Migration of existing I/O

| Pack | Slash NexusTools (keep) | Model schema v1 | Model schema target |
| --- | --- | --- | --- |
| vault / Obsidian | `/obsidian-status`, list/read commands | native `vault_*` | in-process MCP retrace, schema stays NexusTools |
| Zotero | `/zotero-*` | native `zotero_*` | in-process MCP retrace, schema stays NexusTools |
| Overleaf | `/overleaf-*` | native until an MCP exists | stay native (no good MCP yet) |
| Google mail/drive | `/google-*` | native | stay native or official MCP later |
| `web_search` | `/search` | native NexusTool in schema | stay native. Search is ours. Optional MCP wrappers are later. |
| shell / editor / memory / skills / subagent | matching slash | native forever | native forever |

Obsidian today: sophon chat uses Local REST. LM Studio **app** chat can load the same vault via `mcp.json` ([obsidian/README.md](../obsidian/README.md)). The gap is sophon chat not speaking MCP. Closing that gap is this spec, not “Phase 2 in LM Studio only”.

## Policy matching

MCP names are often `server__tool` or bare `tool`. Store rules as `Tool(obsidian.search)` and also honor `Tool(mcp:obsidian/*)`. Deny wins. Plan/chat: MCP **write** tools denied the same way `shell_exec` is. Read MCP tools may stay in the schema.

Circuit breakers do not apply to MCP blobs. MCP filesystem servers that can `rm -rf` need `ask` by default until a path allowlist exists. TODO: map MCP fs roots onto harness workspace.

## Implementation sketch

1. Registry + origin tags on existing `default_chat_tools`. HUD: `NexusTools` label, MCP still 0. No behavior change.
2. MCP host (Python SDK), yaml allowlist, `/tools mcp`, HUD count. Tools join the LM Studio / OpenAI / Anthropic schema through the same loop that already executes NexusTools.
3. Trace: origin, server id, duration, permission decision.
4. Obsidian first, in-process over Local REST. `vault_*` stay in the model schema. MCP rows are inventory + trace aliases.
5. Zotero second. Overleaf stays native.
6. **Next:** stdio transport + one allowlisted scrape/fetch server. Plan: [../../direction/next-mcp-stdio-scrape.md](../../direction/next-mcp-stdio-scrape.md). Umbrella: [../../direction/next-mcp-retrieval-loop.md](../../direction/next-mcp-retrieval-loop.md).

TODO: HF local tool calling still generate-then-display. MCP on HF waits for that backend work. Energy `local` + LM Studio is the first MCP test bed.

## Non-goals

- MCP as a spawn channel ([subagents.md](../direction/subagents.md)).
- Auto-install from the Hub or a plugin marketplace.
- Letting injected LM Studio tools skip `/permissions`.
- Replacing `/search` with a web-search MCP. Native `web_search` stays a NexusTool.
- Exposing Review accept as an MCP tool.

## References

- [MCP specification](https://modelcontextprotocol.io/specification)
- [MCP architecture](https://modelcontextprotocol.io/docs/learn/architecture)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Code MCP](https://code.claude.com/docs/en/mcp)
- [Codex MCP](https://developers.openai.com/codex/mcp)
- [Cursor MCP docs](https://cursor.com/docs/context/mcp)
- [Hermes MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)
- [Obsidian Local REST API](https://github.com/coddingtonbear/obsidian-local-rest-api)
