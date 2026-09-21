# Orchestrator and swarms

## Role

The orchestrator runs **more than one agent** with a schedule, shared context, and explicit state. The human still has one terminal. Workers are subprocesses or server backends, not extra interactive TUIs.

Working name for the loop module: **one**. Backlog path sketch: `one/STATE.md`, free-tier vs default scheduler.

A **subagent** is the inner primitive: one child under a parent turn (spawn, fork, settlement, placement). That seam is specified in [subagents.md](subagents.md). This file is the outer loop: many of those children over time, with `STATE.md` and a human-visible queue. Do not skip the seam and jump to swarms.

## Current behavior

- One chat session. One preload worker in the Textual app.
- Eval and RAG rebuild are slash commands in Chat, synchronous from the user's point of view.
- No swarm UI. Workshop pane was removed. Use Chat + Editor until a swarm overlay exists.
- No spawn tool yet. Local GPU admission is still `/setup` for a single combo.

## Loop formula

```text
Loop = Agents + Schedule + Context + State Control
```

| Part | Meaning | Constraint |
| --- | --- | --- |
| Agents | Model + harness clone | Each agent has a role and tool subset |
| Schedule | Who runs next, retries, budgets | Placement solver in [subagents.md](subagents.md): local VRAM vs API inherit. Outer loop (cron, `STATE.md`) comes after. |
| Context | Shared retrieval, memory session, file snapshots | Do not dump full trees into every worker |
| State control | `STATE.md` + SQLite + job records | Human can stop, pause, reroute. State survives TUI restart |

## Constraints

- **One interactive GPU job.** Local HF / LM Studio / Ollama on this desktop is one resident weight file. Extra children either share those weights (serialized turns), occupy leftover VRAM if the placement solver admits a 2B–4B classify combo, or wait. They do not load a second 12B. Arithmetic and the solver: [subagents.md](subagents.md).
- **Two energy regimes.** Local: VRAM and the 3090 compute stream. API (OpenAI / Anthropic / Google): inherit the parent model by default. Token RPM/TPM and dollars are the cap. Energy is not a local constraint.
- **No fork bomb.** Max workers, max queue, max wall time in `.sophon/` or env (`SOPHON_SWARM_*` / `SOPHON_SUBAGENT_*` to be named when implemented). Depth starts at 1.
- **Isolation.** Workers get a workdir and an allowlist. They do not share the interactive editor buffer. Results return as proposed diffs or reports. Parallel writers use worktrees. That isolates git, not VRAM.
- **Approval boundary.** Merge to git, send mail, Overleaf write: human in the TUI. ResearchAgent never crosses that line. Background children cannot answer 1/2/3 prompts. They fail closed and tell the parent.
- **Observable.** Dashboard tile + Chat `/subagents` then `/swarm status` (names TBD). If it cannot be inspected, it must not run.
- **Headless.** The same scheduler must run from CLI for CI (`sophon-cli swarm` or equivalent, not specified yet).
- **Cost.** Token and request budgets per loop. Free-tier scheduler prefers local models and hard caps.

## Future direction

1. Implement the subagent seam first ([subagents.md](subagents.md) sequence 1–10). Orchestrator calls that API. It does not spawn raw backends.
2. `STATE.md` as the human-readable loop file (goals, blockers, last agent, next action). Machine state in SQLite beside it.
3. Named agents: ResearchAgent first. CodeAgent second (propose only). EvalAgent for `/eval` jobs. Catalog files under `.sophon/agents`.
4. Fan-out: N research reads, 1 synthesizer. Fan-in writes one report buffer in the editor, not N conflicting files. Local fan-out is serialized on the 3090 unless leftover VRAM or API slots exist.
5. Initialization/training/download jobs stay **jobs**, not chat turns (Hub download already has a progress path).
6. Cloud/server control (AWS, B2) is storage/compute for jobs, not a second orchestrator. Credentials in `.env`.
7. Cross-OS: scheduler records OS-neutral job ids. Paths in state files are stored as the runtime root (`SOPHON_WINDOWS_ROOT` vs WSL) explicitly.
8. After spawn works: loop engineering (maker/checker, evidence), then unbounded tool-loop policy, then this swarm overlay.

## Non-goals

- Autonomous 24/7 loops that can spend money or push git without a session flag.
- Hidden workers started by dashboard tiles.
- Rebuilding tmux/screen inside Textual as the isolation mechanism.
