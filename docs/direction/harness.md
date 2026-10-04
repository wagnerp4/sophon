# Harness

## Role

The harness is the only path from a model to the outside world: files, shell, mail, vault, browser-derived data, other agents. Chat is the UX. The harness is the contract.

User-facing copy is **NexusTools** (`/tools`, `/help`). Policy is **sophon harness** (`/permissions`, `/mode`). MCP is a second **origin** for model-facing I/O, not a second policy. Comparison with production CLIs: [harness-comparison.md](harness-comparison.md).

## Agent definition

```text
Agent = Model + Harness
```

A model without a harness can only emit text. A harness without a model is a script runner. ResearchAgent is the first named type on the backlog (search, cite, read vault/Zotero/Overleaf, no unattended write).

## Current behavior

- Tools attach in LM Studio (and other server backends that speak tool calls). HF chat is still primarily generate-then-display.
- Enablement is env-driven capability: `SOPHON_SHELL_TOOLS`, `SOPHON_OBSIDIAN_TOOLS`, `SOPHON_GOOGLE_TOOLS`, `SOPHON_OVERLEAF_TOOLS`, `SOPHON_EDITOR_TOOLS` (default on), TTS/SST flags. Env flags put a tool in the schema. They do not authorize a call.
- Authorization is the harness policy under `.sophon/harness.yaml` (shareable) plus `.sophon/harness.local.yaml` (option 2, gitignored). Merge order: defaults, project yaml, local yaml, in-session grants.
- Rules are Claude-style `Tool` / `Tool(specifier)` with `*`. Evaluation order: deny, then ask, then allow. Compound shell commands (`&&`, `||`, `;`, `|`) must match per subcommand. A grant of `shell_exec(git *)` does not cover `git status && Remove-Item -Recurse -Force C:\`.
- `shell_exec` prompts unless allow-listed. Prompt choices: **1** allow this time, **2** add inferred prefix (`git *`) to the local allow-list, **3** decline (tool result `error: permission denied`). Circuit breakers (root/home wipes, `Format-Volume`, `shutdown`, `Remove-Item -Recurse` on `/` or `C:\`) always prompt.
- `editor_propose_edit` and `memory_propose` prompt when the target is outside the workspace. Inside the workspace they queue for Review. Accept all still writes disk. Option 2 does not skip Review.
- `shell_cd` prompts when the resolved path leaves the workspace (empty/`~` is home).
- Reads (`shell_ls`, `shell_read`, `editor_read`, vault/zotero/overleaf/google) are not prompted inside the workspace.
- Human-typed commands in the editor Terminal tab are not gated.
- `/tools` lists bound packs (today: NexusTools only. MCP count is 0). `/permissions` lists merged rules. `/permissions reload` re-reads yaml. `/mode plan|chat|agent` is session-only permission. `/energy local|api` is the scarce-resource swap ([energy.md](energy.md)). `/compact` is conversation compact, not `/memory compact` ([compact.md](compact.md)). Plan and chat deny `shell_exec`, proposes, and out-of-workspace `shell_cd`. Agent uses the 1/2/3 policy.
- Editor tools: `editor_read`, `editor_propose_edit`, `editor_status`. Disk writes wait for Review (Accept all / Decline all). `/assist status|accept|decline|undo|redo`.
- Shell: one command at a time in the editor Terminal tab and in `shell_exec`. Timeout `SOPHON_SHELL_TIMEOUT_S`.
- Traces: turn traces in chat include permission (allow / ask:once / ask:persist / ask:deny / deny). Session log under `data/chat_logs/` (or `~/.cache/sophon/chat_logs` when the checkout is on `/mnt/c` from WSL).
- The model cannot edit harness yaml through `editor_propose_edit`.

## Constraints

- **Allowlist, not plugin soup.** A new connector is an integration package plus env flag plus `/tools` visibility, or a named row in `.sophon/mcp.yaml`. MCP is a transport for model-facing I/O. It is not a plugin marketplace and not a second undocumented tool channel. Slash-callable behavior stays NexusTools. Spec: [../integrations/mcp/README.md](../integrations/mcp/README.md).
- **No ambient credentials in prompts.** Tokens stay in `.env` or OS stores. Tool schemas must not echo secrets.
- **Interactive vs unattended.** The human TUI session may have shell and editor. Swarm workers default to read tools only.
- **Trace every call.** Tool name, truncated args, duration, error, permission decision. Needed for eval and skill mining.
- **Idempotent propose.** Editor patches must match unique `old_string` (already enforced). Swarm file edits use the same rule.
- **Cancellation.** Esc / stop must abort tool loops (`/tool-rounds`, `/unlimited`). Harness must honor that in agent mode.
- **Policy is not model-writable.** Do not let tools append deny/allow except through the human 1/2/3 prompt.

## Future direction

1. Net allowlist, max subprocesses, max tokens per turn on the same policy object.
2. OS sandbox modes are `read-only`, `workspace-write`, and `full`. Windows `shell_exec` uses WSL bubblewrap for the first two. `full` stays the user process. Arch uses Landlock, then bubblewrap.
3. MCP inventory retrace for Obsidian and Zotero. NexusTools stay in the model schema when names collide. Same permission table. HITL tools stay native. Spec: [../integrations/mcp/README.md](../integrations/mcp/README.md).
4. DeepSeek and other HTTP APIs as **models** under `/energy api`, not as a parallel tool host.
5. Per-agent harness clones: ResearchAgent cannot `shell_exec`. A future CodeAgent can propose edits, not accept them. Spawn, control tools, and GPU/API placement live in [subagents.md](subagents.md).
7. Conversation auto-compact default on (`/compact`). Distinct from `/memory compact`. Spec: [compact.md](compact.md).

## Non-goals

- Letting the model rewrite harness policy.
- Running Docker/K8s as an implicit backend without a dedicated integration spec.
- Merging Toad or other external TUIs as the harness. Toad stays an `--interface toad` host.
- Wrapping Claude Code or Codex CLIs. Sophon is the harness.
