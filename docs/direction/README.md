# Long-term direction

The design objective is a **single terminal surface** that covers software engineering end to end: harness, editor, dashboard, swarm orchestration, and self-evolution. The user stays in one process on Windows, WSL, macOS, or Linux. GUI IDEs and browser consoles remain optional clients, not the source of control.

This folder is the product map. Implementation status lives in [TODO.md](../TODO.md). Runtime install lives in [README.md](../README.md). TUI host choice lives in [TERMINAL.md](../TERMINAL.md).

| Document | Scope |
| --- | --- |
| [VISION.md](../VISION.md) | Thesis, constraints, non-goals, OS contract |
| [harness.md](harness.md) | Tools, permissions, agent loop |
| [subagents.md](subagents.md) | Child spawn, control tools, VRAM/API placement |
| [editor.md](editor.md) | Files, review, languages, completion |
| [dashboard.md](dashboard.md) | Idle telemetry tiles vs work surfaces |
| [orchestrator.md](orchestrator.md) | Swarms, schedule, state, isolation |
| [memory.md](memory.md) | Five-tier memory: tiers, budget, tools, promotion |
| [skills.md](skills.md) | Agent Skills `SKILL.md` catalog, attach, Review create/import |
| [self-evolution.md](self-evolution.md) | Lembas skills, memory, eval gates |
| [os-and-deploy.md](os-and-deploy.md) | Any-OS floor, WSL/Windows split |

## How the pieces combine

```text
Dashboard  ->  what is running, where, at what cost
Editor     ->  what the tree is, what may be written
Chat       ->  plan / chat / agent against one model
Harness    ->  tools + permissions + traces (sophon)
Subagents  ->  child runs + placement (local VRAM vs API tokens)
Orchestrator -> N agents, schedule, STATE.md, isolation
Self-evolution -> skills (Lembas) + memory that only promote after eval
```

Chat without a harness is a chatbot. A harness without an editor cannot land diffs. Subagents without placement oversubscribe the 3090. An orchestrator without a dashboard cannot be supervised. Self-evolution without eval writes untested policy into the loop. The long-term app is the **composition**, not separate products.

## Current vs intended

| Surface | Now | Intended |
| --- | --- | --- |
| Host | Textual, three modes (`Ctrl+D` / `Ctrl+E` / `Ctrl+G`) | Same three modes plus a swarm/status overlay. No fourth "Workshop" pane (removed) |
| Harness | Env-gated tools (shell, vault, Zotero, Google, Overleaf, editor propose/accept) | Policy, traces, skill attach, MCP |
| Subagents | `subagent` / `subagent_fork` in-process one-shot, inherit parent, maxDepth=1 | Named catalog, placement solver, continuable control, leftover 2B / JEV later |
| Editor | Trees + preview + propose/accept changeset | Languages, structural replace, Overleaf write, kernel-less notebooks stay kernel-less until specified |
| Dashboard | Polling tiles from `.env` | Same tiles plus swarm and scheduler health |
| Orchestrator | Single session preload | Loop = agents + schedule + context + state control (`one/`) |
| Skills | `.sophon/skills` + Cursor/Claude scan, `/skill`, `skill_*` tools | Lembas: versioned, reviewed, eval-gated |
| Deploy | WSL source, Windows GPU TUI copy | POSIX-native TUI on Linux/macOS. Windows keeps PE `.venv` |

Working names from the backlog: **sophon** (product and harness), **Lembas** (skills), **one** (loop / `STATE.md`). Package layout in `src/` is a single `sophon` tree. Do not split `lembas/` / `one/` until imports and CLI entry points have a migration plan.
