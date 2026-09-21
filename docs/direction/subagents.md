# Subagents and placement

## Role

A subagent is a **child run** started from a parent turn. It has its own session, a cloned or narrowed harness, and a result that returns to the parent. It is not a second TUI and not yet a swarm.

```text
Subagent  = (prompt, harness clone, isolation, placement)  under a parent turn
Placement = which backend/model/queue slot may run that child
Control   = spawn, send, interrupt, list, wait, settlement notice
```

This document is the plan for **calling and scheduling** children. Loop engineering, unbounded tool loops, and multi-agent orchestration are the next problems. They are named at the end so they do not leak into v1.

## v1 shipped (DSH spawn + fork)

Local in-process one-shot on the resident parent model (12B or the current API parent). JEV is deferred. Continuable send/interrupt and a second GPU weight file are deferred.

| Cap / rule | v1 |
| --- | --- |
| Tools | `subagent` (empty transcript), `subagent_fork` (completed-turn seed) |
| Model | inherit parent backend and weights. Fork rejects a model override |
| Depth | `SOPHON_SUBAGENT_MAX_DEPTH=1` |
| Active | `SOPHON_SUBAGENT_MAX_ACTIVE=1` (serialize the 3090, refuse a second start) |
| Sync | foreground wait, then dispose. No `run_in_background` |
| Plan/chat | spawn tools omitted from the schema and execute-deny |
| `research` | drops `shell_exec`, `editor_propose_edit`, `memory_propose` |
| Child ask | `ask=None` (deny). Scope cannot be widened |
| Human | `/subagents` last run. `/tools` lists spawn tools when enabled |
| Not in v1 | JEV / TypeSafe, leftover-VRAM 2B router, send/interrupt/list, settlement notices |

## Current behavior

- One chat session. One resident local model. `/tool-rounds` caps the **same** session's tool loop.
- `/mode plan|chat|agent` and `.sophon/harness.yaml` apply to that session only. Plan and chat omit spawn tools.
- `/setup` ranks **one** combo for the interactive GPU. It does not admit a second resident model.
- `subagent` / `subagent_fork` run a cloned `_SessionState` on the same backend. Named catalog types beyond `general` and `research` are still backlog.
- OpenAI / Anthropic / Google backends already exist as **models**. They are not a second orchestrator.

Session snapshot used for local arithmetic (2026-09-21, RTX 3090 Ti, 24.0G VRAM):

| rank | backend | model | quant | tok/s | vram | ctx | leftover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hf | gemma4_12b_it | q4 | 12 | 14.3G | 32768 | +9.7G |
| * | lmstudio | google/gemma-4-12b-qat | q4 | 12 | 14.3G | 32768 | +9.7G |
| 3 | lmstudio | qwen/qwen3.8-27b | server-gguf | 5 | 19.2G | 8192 | +4.8G |
| 4 | hf | qwen3_5_2b | q4 | 72 | 2.4G | 32768 | +21.6G |
| 7 | hf | qwen3_5_4b | q4 | 36 | 4.8G | 32768 | +19.2G |
| 8 | hf | gemma4_e2b_it | q4 | 29 | 6.0G | 32768 | +18.0G |

`/setup` recipe: local before download, then `2*tok/s + 5*log10(ctx) + 30*local + 15*tools[agent] - 10*n_deps`, 10% VRAM headroom, CPU offload -100 after the hard filter.

## Split from adjacent docs

| Layer | Owns | Doc |
| --- | --- | --- |
| Harness | Tools, permissions, traces for **one** agent | [harness.md](harness.md) |
| Subagent | Child start, control tools, placement admission | this file |
| Orchestrator | N agents over calendar/queue, `STATE.md` | [orchestrator.md](orchestrator.md) |
| Loop engineering | Recurring maker/checker with evidence gates | next plan |
| Skills | Procedures the child may attach | [skills.md](skills.md) |

`Agent = Model + Harness` stays true. A subagent is another agent. The scheduler decides whether that second agent may occupy VRAM, an API slot, or must wait.

## What the competition does

Vault index: [Harness/About](obsidian://open?vault=Obsidian%20Notes&file=KB%2FIndex%2FFormal%20Sciences%2FComputer%20Science%2FSoftware%20Engineering%2FAI%2FAgentic%2FHarness%2FAbout.md). Local trees: `Computer Science/AI/Agentic/Harness/{claude-code,codex,deepseek-harness,harness-foundry}`.

### Shared contract

Every production CLI converges on the same shape:

1. **A model-facing spawn tool** (Claude `Agent`/`Task`, Codex `spawn_agent`, DSH `subagent` / `subagent_fork`).
2. **Context choice**: fresh child (spawn) vs seed from parent turns (fork). Forks keep the parent prefix for cache reuse. Fresh children need a standalone prompt.
3. **Sync choice**: wait for the result, or return an id and continue. Continuable children default to **background**.
4. **Control tools**: send, interrupt, list. Codex also has `wait_agent` and `followup_task`. DSH tells the parent when a child settles instead of asking it to poll.
5. **Caps**: depth, live children, token/tool budgets. DSH `maxDepth` defaults to **1**. `maxActiveSubagents` defaults to **8**. Foundry stacks cap tool calls (10 on triage, 40 on implementer).
6. **Permission freeze**: the child cannot widen scope. DSH injects that as a fixed child statement. Claude clones a tool allowlist from agent frontmatter. Sophon already plans per-agent harness clones.
7. **Result, not transcript**: the parent sees a final answer or a settlement notice. Intermediate child tool steps stay in the child session.

### Claude Code

- Catalog of **named types** (`general-purpose`, `Explore`, plugin agents). Each agent is markdown + YAML: `name`, `description` with trigger examples, `model` (`inherit` / sonnet / opus / haiku), `tools`, optional `isolation: worktree|remote`.
- The parent picks `subagent_type`. The runtime may override model. Explore used to force Haiku, then switched to inherit (capped at Opus) because the small model was the wrong specialist for grep-heavy work.
- Fork (`subagent_type: "fork"`) inherits conversation and prompt cache. Forks do not accept a different model.
- Interactive non-teammate spawns run in the **background by default**. Continue with `SendMessage`, not a `resume` parameter.
- Isolation is a **worktree** (disk), not a second GPU. Parallel writers do not share the parent's dirty tree.
- Hooks: `agent.spawn` before start, SubagentStop when the child wants to halt. Children cannot escalate permissions.

### Codex

- Collaboration namespace: `spawn_agent`, `followup_task`, `send_message`, `wait_agent`, `interrupt_agent`, `list_agents`. Not hidden inside `functions.exec`.
- Default prompt treats **all agents as equally capable with the same tools**. Specialization is optional via `agent_type` / role, which can **lock** model and reasoning.
- `fork_turns`: `"all"` | `"none"` | integer. Full-history forks inherit parent model and reject overrides (same KV-cache reason as DSH).
- Modes: explicit-request-only vs proactive parallelization. Nested spawn is allowed in the prompt. The TUI groups children by worktree and thread.
- Guardian / compact / memory-consolidation are **internal** subagent sources, not user-spawned workers.

### DeepSeek Harness

This is the most complete **seam**. Service + named providers + model-facing tools.

| Piece | Role |
| --- | --- |
| `dsh-subagent` | Registry, depth, live-child pool, settlement |
| `subagent-spawn-in-process` | Fresh child, empty transcript |
| `subagent-fork-in-process` | Seed completed parent turns |
| `subagent-acp` / `*-codex` / `*-claude-code` | Out-of-process children |
| `dsh-tool-subagent` | `subagent` tool, optional child `provider`/`model` |
| `dsh-tool-subagent-control` | `send_message`, `interrupt_agent`, `list_agents` |

- One-shot vs continuable. Continuable: start returns `started subagent <id>`, later a **settlement notice** (`Background subagent <id> finished...`). Do not poll.
- Child LLM selection is an **allowlist of exact routes**, sampled once per top-level session. Fork tools cannot change the route.
- `toolFilter` / `persona` / `maxDepth` are **per tool instance**, not free-form per call. Another specialist means another named tool or another agent type.
- Admission refuses at capacity (`ACTIVATION_LIMIT_REACHED`). It does not queue a parent behind its own occupied slot.
- Publication is the ownership boundary: failed start rolls back, success transfers the run to the caller.

Sophon should copy this seam, not wrap DSH.

### harness-foundry

Foundry is **not** a subagent runtime. It is the taxonomy for **what a child is allowed to be**:

| Layer | Question |
| --- | --- |
| L1 Interface | Which model |
| L2 Composition | Tools, skills, `STATE.md` |
| L3 Execution | Sandbox, token budget, tool-call cap |
| L4 Reliability | Trace, recover, evidence |

Stacks are scheduled recipes. `triage` uses a smaller API model, 50k tokens, 10 tool calls, grep-only. `implementer` uses Sonnet, 100k tokens, 40 tool calls, worktree write, revert-on-test-fail. That is the correct reading of "small expert": **narrow stack**, not "always a 2B local".

### Cursor (extra data point)

Cursor's `Task` tool is the same family: typed subagents, optional model slug, isolated git worktrees for parallel writers, read-only explorers, and a parent that must not duplicate the child's work. Treat it as confirmation of the contract, not as a dependency. Sophon does not wrap Cursor.

## Local 3090: concurrent weights

Keep the interactive parent on the current 12B QAT (~14.3G at 32k). Leftover on the card is about **9.7G** before fragmentation.

| Second resident | q4 VRAM | Fits beside 12B? |
| --- | --- | --- |
| qwen3_5_2b | 2.4G | one, maybe two if KV stays small |
| qwen3_5_4b | 4.8G | one |
| gemma4_e2b_it | 6.0G | one, tight |
| second 12B | 14.3G | no |
| qwen 27B | 19.2G | no |

Allocator fragmentation, LM Studio already holding the 12B, and KV growth all shrink that leftover. Treat **one extra 2B–4B** as the optimistic concurrent case, not three experts in parallel.

## Should children be 2B / E2B experts?

**Sometimes, after admission, for bounded jobs. Not as the default spawn path.**

Reasons to keep the parent model:

- Claude's Explore regression: a cheaper specialist failed the actual job (repo grep + synthesis). Inherit is the safe default.
- Codex and DSH default to inherit. Route change is opt-in and forbidden on full forks.
- A 2B that cannot use tools reliably is not an expert. It is a classifier.
- Loading a second weight file while the 12B is resident is the scarce resource. Context isolation is not.

Where a small local **is** the right specialist:

| Job | Why a 2B–4B can hold it | Stack shape |
| --- | --- | --- |
| Route / classify the next task type | Short prompt, no tools or one tool | Foundry `triage`: tiny budget, 10 calls |
| Summarize a child result into the parent | Read text, emit a paragraph | no shell |
| Fill a JSON schema (todo list, file list) | Structured decode | no write |
| Draft-model / speculative decode | Token draft, parent verifies | not a subagent |

Where it is the wrong specialist:

- Implement a patch across files
- Review a large diff
- Multi-hop vault/Zotero research
- Anything that needs the parent's tool pack at full quality

Eval gate before a 2B is in the default catalog: run the same fixture on 12B vs 2B (`/eval` later, skill-level fixture until then). If the 2B's artifact fail rate is higher, it does not ship as that `agent_type`.

## Options besides smaller experts

These save RAM/VRAM **without** a second weight file. Prefer them in this order on the 3090.

### 1. Same weights, many harness clones (default local)

N children can share the **resident** 12B. They differ by system prompt, tool allowlist, and session id. The GPU runs one forward at a time. The scheduler serializes or time-slices turns. This is the only local fan-out that does not fight `/setup`.

Cost: latency, not VRAM. Benefit: quality stays at the parent floor.

### 2. Spawn vs fork (context isolation)

- **Spawn**: empty child transcript, standalone prompt. Parent context is not copied. This is the main way to keep the parent's window small.
- **Fork**: copy completed turns, keep prefix cache. Do not change model.

Isolation of **memory** is not isolation of **VRAM**. Both still use the resident weights unless placement says otherwise.

### 3. Queue, do not oversubscribe

`maxActiveLocal = 1` for HF/LM Studio on this machine. Extra spawn requests become `queued` until the GPU slot is free. DSH refuses at capacity rather than deadlock the parent. Sophon should **queue** one-shot children and **refuse** continuable children that would pin a second Activation on the same GPU.

### 4. Sequential swap

Park or unload the parent, load the child combo, run, restore the parent. Use only when the child **must** be a different local model and will not fit in leftover VRAM. Swap cost is seconds to tens of seconds. Not for a 2B classify call.

### 5. Worktree isolation (disk)

Claude/Codex/Foundry isolate **writers** with git worktrees. That prevents merge fights. It does not create VRAM. ResearchAgent and other read-only types stay in the workspace.

### 6. Hybrid energy

Parent stays local (interactive, private corpus). Children that need fan-out go to OpenAI / Anthropic / Google. Reverse is valid too: API parent, local child for vault-only reads. Credentials stay in `.env`.

### 7. Prefix / KV reuse

Forks and repeated system prompts should keep a stable prefix so llama.cpp / LM Studio prefix cache hits. Do not inject per-child timestamps into the system prompt.

### 8. Do not spawn

A skill attach plus a tool filter on the **current** session is cheaper than a child. Use a subagent when the work would pollute the parent window or must run in the background.

### 9. CPU offload (last)

`/setup` already scores CPU offload at -100. Allowed as a human override, not as automatic admission.

### 10. Second GPU / remote host (not this machine)

Out of scope until a device other than this 3090 exists. The placement object should still have a `device` field so the code does not assume one card forever.

## Two energy regimes

The user statement is the policy:

> API models: reuse the same model. Energy is paid in the bill. Local models: energy and VRAM are the constraint.

| Regime | Scarce resource | Default child model | Concurrency limiter |
| --- | --- | --- | --- |
| `api_token` | dollars, RPM, TPM | **inherit parent** | `SOPHON_SWARM_MAX_API`, token budget |
| `local_vram` | 24G, one compute stream | **inherit resident weights** | `maxActiveLocal=1`, leftover VRAM for an extra combo |
| `hybrid` | both | inherit unless the solver upgrades/downgrades | per-leg limiter |

API may still pick a cheaper **named** stack (Foundry triage / Claude Haiku) when the agent type is `classify` or `summarize` **and** eval has accepted that pair. That is cost, not energy. It is optional. Inherit remains the default.

## Adaptive placement

The parent proposes a **task descriptor**. The scheduler **admits, rewrites, or queues**. The parent does not pick VRAM.

### Task descriptor (parent -> scheduler)

```text
kind:        explore | classify | implement | review | synthesize | fetch
agent_type:  research | code | eval | general
context:     spawn | fork
isolation:   session | worktree | none
sync:        foreground | background
tools:       inherit | allowlist
quality:     inherit | low | mid | high
budgets:     max_tokens, max_tools, max_wall_s
prompt:      standalone text (required for spawn)
```

`kind` is the job. `agent_type` is the harness clone. They are not the same. Explore+research is read-only. Implement+code may propose edits. Classify+general should be a tiny stack.

### Heuristic router (v1, no extra model)

Same style as Adaptive-RAG: deterministic text policy, then a model only if the heuristic is uncertain.

| Signal | Placement |
| --- | --- |
| Write / patch / `editor_propose` | `code`, inherit quality, worktree if parallel, **no** 2B |
| Review / diff / "check this" | `review`, inherit or API, read tools |
| Search / cite / vault / Zotero | `research`, read tools, spawn (fresh context) |
| Short label, JSON, yes/no, "which files" | `classify`, low quality, may use 2B if resident leftover fits **or** API mini |
| "summarize the child" | parent session, do not spawn |
| Needs parent conversation | fork, inherit model, no route change |
| Independent and parallel | background spawn, queue on local |

If two signals conflict, pick the **higher** quality floor. Never downgrade `implement` to 2B because VRAM is tight. Queue or escalate to API instead.

Optional later: a 2B router **only** when the heuristic returns `uncertain`. That router must not have shell tools. **Deferred with JEV.** A TypeSafe/OpenRouter round-trip would add a key, latency, and an offline failure mode without changing the legal set while `maxDepth=1` and inherit is the only combo. Keep a Choice-shaped interface in this doc. Do not add `typesafe-sdk` until `|legal| > 1` and the heuristic is actually uncertain.

### Cost model (solver)

Extend `/setup`'s `SetupCombo` with a **pack**:

```text
score_pack = sum_i score(combo_i)
             - 1000 * over_vram
             - 50 * n_swaps
             - 20 * queue_wait_s
             + 15 * tools_ok
             + quality_bonus(kind, params_b)
```

`quality_bonus` is a table, not a learned policy in v1:

| kind | min params (local q4) | API default |
| --- | --- | --- |
| classify | 2B | inherit or mini |
| summarize | 2B | inherit |
| explore / fetch | 12B (or E4B after eval) | inherit |
| review / implement / synthesize | 12B | inherit |

Admission:

1. If `api_token` or `hybrid` with keys: inherit parent route unless the table names a cheaper API id **and** the agent type allows it.
2. If `local_vram` and child combo == resident combo: admit, serialize GPU.
3. If leftover VRAM >= child.need + 10% headroom **and** kind allows that size: admit second resident (at most one extra).
4. Else if kind requires a different local size: queue or swap (swap only with explicit policy).
5. Else: refuse with a machine-readable reason (`VRAM`, `DEPTH`, `ACTIVE`, `KIND_QUALITY`).

The solver is **not** another agent. It is code. Traces record the decision.

### Fairness and pinning

- Continuable children pin a control slot (`maxActiveSubagents`) even when idle. On local GPU, continuable + extra weights is the failure mode. v1: continuable children **must** share resident weights.
- One-shot children may swap or use leftover VRAM.
- Nested spawn: `maxDepth=1` until loop engineering. The child tool is visible but start rejects at depth 1 (DSH pattern).

## Control surface

Model-facing tools (names TBD, DSH-shaped):

| Tool | Does |
| --- | --- |
| `subagent` | Spawn or named-type start. `run_in_background` default true for continuable, false for one-shot. |
| `subagent_fork` | Seed completed parent turns. No model override. |
| `send_message` | Steer a live child, or start a turn on idle. Returns acceptance id, not a reply. |
| `interrupt_agent` | Stop current child turn. Keep inbox. |
| `list_agents` | Direct children. Status: `running` / `idle` / `queued` / `ready`. |

No parent-side `wait` poll in v1. Runtime injects a settlement notice into the parent stream (DSH). Codex `wait_agent` can wait until a later plan if notices prove insufficient.

Human:

- `/subagents` status (durable id, type, model, vram slot, state)
- `/mode` still gates whether spawn exists (plan: no spawn execute)
- Dashboard later: one tile for GPU slot + API in-flight. Not a fourth pane.

Caps (env / `.sophon/harness.yaml`, names to lock at implement):

```text
SOPHON_SUBAGENT_MAX_DEPTH=1
SOPHON_SUBAGENT_MAX_ACTIVE=1
SOPHON_SUBAGENT_MAX_LOCAL=1
SOPHON_SUBAGENT_MAX_TOKENS
SOPHON_SUBAGENT_MAX_TOOLS
SOPHON_SUBAGENT_MAX_WALL_S
```

Child harness: merge parent policy with a **narrower** overlay. Deny wins. Child cannot write harness yaml. Child cannot `/permissions` persist. Same 1/2/3 prompt is **not** shown inside a background child. Risky calls fail with "ask the parent". Interactive parent remains the only HITL surface.

## Named catalog (v1 agents)

Markdown under `.sophon/agents/<type>/AGENT.md`, same idea as Claude plugin agents and Sophon skills. Frontmatter: `name`, `description`, `model` (`inherit` | preset | `openai:…`), `tools`, `quality`, `isolation`.

| type | tools | model default | isolation |
| --- | --- | --- | --- |
| `general` | inherit session mode | inherit | session |
| `research` | read: vault, zotero, google, overleaf, editor_read. no shell_exec | inherit | session |
| `code` | editor propose, shell as parent policy | inherit | worktree if parallel |
| `eval` | no writes. `/eval` job path when wired | inherit | session |
| `classify` | no shell, no editor write | inherit, scheduler may pick 2B/API mini | session |

ResearchAgent stays the first named type on the product backlog. It is a catalog row, not a special case in the spawn tool.

## Implementation sequence

Stop after each step. Do not start loop engineering in the same change.

1. **TODO:** `TaskDescriptor` + `PlacementDecision` types. Pure functions over `SetupCombo` and device snapshot. Unit-test the 3090 table (12B resident + 2B leftover / 27B reject / classify-vs-implement).
2. **TODO:** Agent catalog loader (`.sophon/agents`, scan like skills). No extra spawn types yet.
3. **DONE:** One-shot `subagent` tool, same backend as parent, foreground wait, `maxDepth=1`, result = child final text. Plan/chat: tool absent and execute-deny.
4. **TODO:** Background one-shot + settlement notice + `list_agents` / `interrupt_agent`. Local GPU still serial.
5. **DONE:** `subagent_fork` with no model override.
6. **TODO:** Placement rewrite: leftover VRAM may load one 2B/4B for `classify` only. Swap path behind a flag. JEV Choice deferred until the legal set is larger than inherit.
7. **TODO:** API inherit + optional cheaper classify route. Token/RPM caps.
8. **TODO:** Continuable children + `send_message`. Local continuable = shared weights only.
9. **TODO:** Worktree isolation for parallel `code` children. Review still lands in the parent editor.
10. **DONE:** `/subagents` last-run line. Dashboard occupancy still later.

Slash help: new `subagents` section next to `tools`.

## Out of this plan

Named so the next spec has a hook. Do not implement here.

- **Loop engineering** — cron / `/loop`, maker then checker, evidence, L1–L3 autonomy. Foundry `with-outerloop` and the vault [Loop/About](obsidian://open?vault=Obsidian%20Notes&file=KB%2FIndex%2FFormal%20Sciences%2FComputer%20Science%2FSoftware%20Engineering%2FAI%2FAgentic%2FLoop%2FAbout.md) page. Needs this spawn seam first.
- **Infinite tool calling** — `/tool-rounds unlimited` already exists for one session. Multi-child loops need per-child caps, stop hooks that look at artifacts (not self-grade), and cancellation that walks the tree.
- **Multi-agent orchestration** — `STATE.md`, fan-in synthesizer, swarm overlay, headless `sophon-cli swarm`. That is [orchestrator.md](orchestrator.md). Subagents are the primitive it will call.

## Non-goals

- Wrapping Claude Code, Codex, or DSH as the child runtime.
- Letting the parent model pick arbitrary Hub ids.
- Concurrent 12B+12B or 12B+27B on this 3090.
- Children that persist allow-list grants.
- A learned RL scheduler in v1.
- Docker/K8s as implicit isolation.
- Treating MCP as a second spawn channel.

## Sources

Harness map and study order:

- [KB Harness/About](obsidian://open?vault=Obsidian%20Notes&file=KB%2FIndex%2FFormal%20Sciences%2FComputer%20Science%2FSoftware%20Engineering%2FAI%2FAgentic%2FHarness%2FAbout.md)
- [ai-boost/awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering)
- [anthropics/claude-code](https://github.com/anthropics/claude-code)
- [openai/codex](https://github.com/openai/codex)
- [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)
- [cobusgreyling/harness-foundry](https://github.com/cobusgreyling/harness-foundry)

Local package notes used for the seam:

- `deepseek-harness/packages/subagent/subagent/README.md`
- `deepseek-harness/packages/subagent/tool-subagent/README.md`
- `deepseek-harness/packages/subagent/tool-subagent-control/README.md`
- `codex/codex-rs/prompts/src/model_messages/multi_agent.rs`
- `claude-code/plugins/plugin-dev/skills/agent-development/SKILL.md`
- `harness-foundry/SPEC.md`, `stacks/implementer/stack.yaml`, `stacks/triage/stack.yaml`
