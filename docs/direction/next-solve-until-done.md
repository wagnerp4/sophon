# Plan: solve-until-done (no cloud GPU fantasy)

Umbrella: [next-mcp-retrieval-loop.md](next-mcp-retrieval-loop.md). Outer loop context: [orchestrator.md](orchestrator.md). Inner spawn: [subagents.md](subagents.md). Energy: [energy.md](energy.md).

## Goal

A **local + API** outer loop that keeps working a task until an evidence gate says done (or budget / human stop). Not a remote GPU swarm, not bounties, not free-tier marketplace fillers.

## Locks

1. One interactive GPU job remains the law. Extra local children serialize or use leftover-VRAM rules from [subagents.md](subagents.md). API children inherit `/energy api` and the spend ledger.
2. Evidence is external: tests green, files exist, checklist boxes in `STATE.md`, harness traces. The maker model does not pass its own “I am done” claim.
3. Human can always `/stop`, Esc, or decline Review. Background children fail closed on 1/2/3.
4. Round and dollar caps stay real: `/tool-rounds`, `SOPHON_TOOL_MAX_ROUNDS`, energy daily cap + ask-and-hold. “Until done” never means unlimited spend.
5. No pod/cluster ordering. No Kaggle/Colab scheduler. No bounty price → rent GPUs.
6. Package layout stays one `sophon` tree. Working name **one** for loop files is fine. Do not split `src/one` until a migration plan exists.

## Shape

```text
goal (human)
  -> STATE.md (goal, blockers, next, evidence)
  -> maker turn (agent mode, tools, optional subagent)
  -> checker (deterministic or cheap API/local classify)
  -> update STATE.md + traces
  -> repeat until pass | cap | human stop
```

Checker v1 prefers **code**: exit codes, path exists, regex on logs. Model-as-judge is optional and never the only gate for merge/push.

## Substeps

1. **`STATE.md` contract**
   - Path: `.sophon/STATE.md` (session/project) or path from `/loop path`.
   - Sections: Goal, Constraints, Evidence, Blockers, Next, Log (append-only short lines).
   - Machine mirror optional later (SQLite). Human-readable file is enough for v1.
   - Slash: `/loop status` prints Goal / Next / last evidence. `/loop clear` only with confirm.

2. **`/loop` entry**
   - `/loop start [goal text]` writes Goal + Next, switches or reminds agent mode.
   - `/loop once` runs one maker→checker cycle and stops.
   - `/loop until` cycles until pass, round budget, or energy hold.
   - `/loop stop` sets a flag the cycle honors between steps.

3. **Maker**
   - Reuses current agent tool loop. No second harness.
   - May call `subagent` / `subagent_fork` within existing depth/active caps.
   - Must write or propose evidence paths into STATE (via editor Review when disk write is gated).

4. **Checker**
   - Config block in `.sophon/loop.yaml` (or STATE frontmatter): list of checks.

```yaml
checks:
  - type: path_exists
    path: "src/integrations/mcp/host.py"
  - type: shell
    command: "python -m compileall src/integrations/mcp"
    ask: true
  - type: rag_probe_min_hits
    query: "MCP stdio"
    min_hits: 1
```

   - `shell` checks go through the same `shell_exec` policy (ask by default).
   - Fail → append Blockers + Next. Pass → mark done, stop `/loop until`.

5. **Budgets**
   - Max cycles: `SOPHON_LOOP_MAX_CYCLES` (default small, e.g. 8).
   - Inherit tool-round cap per maker turn.
   - Energy ask-and-hold aborts `/loop until` cleanly and records reason in Log.

6. **Observability**
   - Dashboard or chat footer: `Loop: idle|running|held|done` + cycle i/N.
   - Traces keep maker/checker ids. Headless: `sophon-cli loop once` later (same contract). TUI-only is incomplete per VISION.

## Done when

- Human can `/loop start`, `/loop once`, and see STATE.md update.
- A failing check continues. A passing check stops `/loop until`.
- Caps and Esc interrupt without orphan workers.
- No new cloud infra. Works under `/energy local` and `/energy api`.

## Out

- Swarm overlay / N-agent fan-in (orchestrator.md sequence after this)
- Continuable remote agents / shared knowledge layer
- Self-evolving skills promotion inside the loop (needs [self-evolution.md](self-evolution.md) eval gates)
- Unlimited tool policy as the default
- Bounty marketplace UX

## Relation to vault wishlist

Vault “Loop Engineering” and “Bounties” collapse here to: **evidence-gated retry on the machine you already pay for** (local VRAM or API dollars). The rest stays backlog.
