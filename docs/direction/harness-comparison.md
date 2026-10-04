# Differences that matter

Sophon is an allowlist CLI with Claude-shaped permissions and a DeepSeek-shaped spawn/fork seam. Nexus is the chat persona. The harness is not a plugin host, not an IDE, and not a wrapper around Claude Code or Codex.

Production harnesses share a contract. They disagree on who hosts the loop and which resource the scheduler treats as scarce. This note is the durable cut of that comparison. Product map: [README.md](README.md). Policy: [harness.md](harness.md). Spawn: [subagents.md](subagents.md).

Three workstreams from this note are specified separately: [energy.md](energy.md), [compact.md](compact.md), [../integrations/mcp/README.md](../integrations/mcp/README.md). Iteration plan: [next-energy-mcp-compact.md](next-energy-mcp-compact.md).

## Shared contract

Every production CLI converges on:

1. A model-facing spawn tool.
2. Spawn (empty child transcript) vs fork (seed completed parent turns).
3. A permission freeze. The child cannot widen scope.
4. A result back to the parent, not the child transcript.
5. Caps on depth and live children.
6. Skills as `SKILL.md` or the same idea.
7. Writes through some policy.
8. Session state that can be reconstructed.

Sophon already has a first slice: `Tool(specifier)` deny/ask/allow, plan/chat/agent, `subagent` / `subagent_fork` with inherit + `maxDepth=1` + `maxActive=1`, Agent Skills catalog, CoALA memory with Review promotion.

## Three hosts

| Host | Who owns the loop |
| --- | --- |
| Sophon / nexus | One Textual process: dashboard + editor + chat. New connector = package + env flag + `/tools`. Wrapping Claude Code, Codex, or DSH is a documented non-goal. |
| DeepSeek Harness | Cordis. Everything is a plugin, including the agent loop. Profiles compose web / headless / SDK / ACP. Subagent providers can be in-process or another product. Session log is the source of truth: model-visible means logged. |
| Claude Code, Codex, Cursor | Vendor CLI or IDE owns the loop. Skills, MCP, Task/Agent, worktrees, and compaction exist because the host already has an editor, git, and a cloud backend. |
| Hermes | Personal agent with a TUI and messaging gateways. Learning loop, cron, seven terminal backends. Adjacent to a lab coding TUI, not a substitute for one. |

Codex is the closest architecture peer (Rust core, not plugin soup). Claude Code is the closest policy peer (`Tool(git *)`, 1/2/3). DSH is the closest spawn-seam peer. Cursor is the closest editor HITL peer. Hermes is the closest memory/skills-from-experience peer, which sophon refuses to copy as an ungated write path.

## Permissions vs sandbox

Sophon copied Claude’s rule language and HITL prompt. It did not copy Codex’s OS sandbox (seatbelt / landlock / workspace-write). Shell still runs in the user process. Circuit breakers always prompt. They do not confine argv. DSH and Hermes can swap an execution world (`ctx.sandbox`, Docker/SSH/Modal). That is a real gap. OS sandbox stays on the [harness.md](harness.md) future list. It is not in the first three workstreams.

## Subagents

v1 is DSH’s in-process one-shot with the 3090 as the limiter: inherit parent weights, foreground wait, `maxActive=1`, no send/interrupt/list, no settlement notice, no worktrees. Claude, Codex, and Cursor default continuable children to background, isolate writers in git worktrees, and run several at once because energy is an API bill. DSH’s default live pool is 8. Sophon serializes the GPU on purpose.

Named types stop at `general` and `research` (research drops `shell_exec` and proposes). Not in the first three workstreams.

## Tools

Sophon is native lab packs (vault, Zotero, Overleaf, no-key academic search) rather than MCP-first. MCP is currently a zero in the HUD (`Skills: N · Tools: N · MCP: 0`) and a TODO in `harness_inventory_counts`. Claude, Codex, Cursor, and Hermes treat MCP as a first-class channel. Cursor also has browser/Task as product surface. Hermes has a large toolset plus Python RPC.

HF chat is still mostly generate-then-display. Tool calling is on LM Studio and other server backends.

Intended split: MCP-first for model-facing I/O connectors, NexusTools for the `/command` family and HITL. Spec: [../integrations/mcp/README.md](../integrations/mcp/README.md).

## Skills and memory

Catalog shape matches Claude/Cursor/Hermes (`SKILL.md`, progressive disclosure). Sophon scans Cursor and Claude trees read-only and will not let the model write skills or facts. Hermes creates skills from experience, searches past sessions, and models the user. Sophon’s [self-evolution.md](self-evolution.md) wants eval gates before anything like that lands. Claude’s memory tool lets the model edit a directory. Sophon stops the model at scratch.

Not in the first three workstreams.

## Context

No conversation auto-compact. Window = trailing turns + a budgeted memory pack + `/tool-rounds`. Claude, Codex, Cursor, and Hermes all compact. DSH projects history from an append-only log. Sophon traces permission decisions in the turn log. It is not DSH’s “if it reached the model, it is in the log” invariant.

`/memory compact` collapses scratch notes into an episode. That is not conversation compact. Spec: [compact.md](compact.md).

## Hooks

None. Claude’s PreToolUse / PostToolUse / SubagentStop and DSH’s `agent/*` + `tools/*` waterfalls are how those products extend the loop without forking it. Sophon extends by shipping another integration package. Hooks stay later. MCP does not become a hook bus.

## Writes

`editor_propose_edit` queues Review. Option 2 on a permission prompt does not skip Review. Policy yaml is not model-writable. Claude and Codex write under sandbox/permission. Cursor applies in the IDE. The Review gate stays. MCP servers do not accept diffs.

## Local vs API

Claude Code and Codex assume the scarce resource is dollars and rate limits. Cursor assumes a hosted model picker. Sophon currently assumes one resident 12B and leftover VRAM, while OpenAI / Anthropic / Google backends already exist as `/backend` and `/model openai:NAME`. Those are a picker, not an energy regime.

Intended: `/energy local|api` as a mode swap orthogonal to `/mode plan|chat|agent`. Spec: [energy.md](energy.md).

## Intentional absences

Not wrapping Claude Code / Codex / DSH as the child runtime. No plugin marketplace. No messaging gateway. No model-writable harness yaml. No concurrent 12B+12B. Hermes learning-loop is adjacent to self-evolution and is gated on eval.

## Unfinished relative to the shared contract

OS sandbox, worktrees, background children + settlement, MCP, conversation compact, hooks, named catalog, HF tool calling. First three to specify for implementation: energy mode, MCP + NexusTools, auto-compact.

## References

- [anthropics/claude-code](https://github.com/anthropics/claude-code)
- [Claude Code docs](https://code.claude.com/docs/en/overview)
- [openai/codex](https://github.com/openai/codex)
- [Codex documentation](https://developers.openai.com/codex)
- [Codex sandbox and approvals](https://developers.openai.com/codex/security)
- [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)
- [DeepSeek Harness docs](https://deepseek-harness.github.io/deepseek-harness/)
- [DSH architecture](https://github.com/deepseek-ai/deepseek-harness/blob/main/docs/architecture.md)
- [Cordis (arXiv:2608.25512)](https://arxiv.org/abs/2608.25512)
- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- [Hermes Agent docs](https://hermes-agent.nousresearch.com/docs/)
- [Agent Skills specification](https://agentskills.io/specification)
- [ai-boost/awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering)
- [cobusgreyling/harness-foundry](https://github.com/cobusgreyling/harness-foundry)
- [CoALA (arXiv:2309.02427)](https://arxiv.org/abs/2309.02427)
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
