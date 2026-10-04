# Conversation compact (`/compact`)

Production harnesses compact the **conversation body** when the window fills. Sophon does not. The window is trailing turns plus a budgeted memory pack plus `/tool-rounds`.

`/memory compact` is a different verb. It folds scratch notes into one episodic row. Do not overload it.

Default: auto-compact **on in local energy**. In API energy it is **off** unless `/compact billed on`, because the summary is another hosted completion.

## What gets compacted

Compacted:

- Older `user` / `assistant` turns in `state.messages`.
- Older tool-result payloads that are already in that list.

Rebuilt every turn, never summarized:

- System prompt, harness hint, skills catalog, memory pack, RAG block.

Those blocks already have char budgets ([memory.md](memory.md), [skills.md](skills.md)). Compacting them would fight those budgets.

Keep a tail of recent turns verbatim (`SOPHON_COMPACT_KEEP_TURNS`, default 6, same order of magnitude as `SOPHON_MEMORY_RECALL_TURNS`).

## Slash surface

```text
/compact              status (on/off, last run, tokens before/after if known)
/compact on           enable auto (default)
/compact off          disable auto. Manual /compact now still works.
/compact now          run once, even if under the threshold
/compact status       same as bare /compact
/compact billed on    allow auto-compact while energy is api (billed)
/compact billed off   API auto-compact stays off (default)
```

`/help generation` (or a `context` section) lists it. Do not put this under `/memory`.

Env:

```powershell
$env:SOPHON_COMPACT="1"
$env:SOPHON_COMPACT_KEEP_TURNS="6"
$env:SOPHON_COMPACT_RATIO="0.70"
```

`SOPHON_COMPACT=0` disables auto at process start. `/compact off` is the session override.

## Trigger

Before assembling the next model request, if auto is on and estimated conversation tokens (messages only, not system blocks) exceed `ratio * context_length`, run compact.

`context_length` comes from the loaded combo (`/setup` ctx, LM Studio reported n_ctx, or API model table). If unknown, fall back to 32k for local and 128k for API.

Do not compact in the middle of an in-flight tool loop. Wait for the turn to settle, then compact before the next user message or before the next parent request after a one-shot subagent returns.

## Algorithm (v1)

1. Split `state.messages` into `prefix` (older than keep) and `tail` (last `KEEP_TURNS` user/assistant pairs, including their tool messages).
2. If `prefix` is empty, no-op.
3. Summarize `prefix` with the **current** model into one assistant-visible system or `user` summary message labeled `Conversation summary (compacted):`. The summary must list decisions, file paths, and open TODOs. It must not invent tool results.
4. Replace `prefix` with that one message. Keep `tail`.
5. Append a compact event to the chat log / turn trace: timestamp, n_removed, n_kept, chars before/after.
6. Print a one-line HUD note: `(compacted N turns -> summary)`.

Failure (model error, empty summary): leave messages unchanged, disable auto for the rest of the session, tell the human. Do not retry in a loop.

Local energy: auto-compact on. Same resident weights. HUD `(compacted N turns)`. Not a USD event.

API energy: auto-compact off unless `/compact billed on`. Skip line must say why, with an estimate: `(compact skipped: api, est $0.02 — /compact now or /compact billed on)`. `/compact now` prints the estimate, ask-and-hold, then runs. Ledger row `purpose=compact`. Over cap: skip. Never silent spend. Spec: [energy.md](energy.md).

## Relation to memory

After a successful compact, optional: write the summary into working scratch (`memory_write` path) so `/memory compact` can later promote it. v1 does **not** auto-promote to `facts.md`. Review still owns durable facts.

Do not treat compact as DSH session-log projection. v1 mutates the in-memory message list and writes a trace line. A later “model-visible means logged” pass can lift the compact event into an append-only log. TODO: persist compact generations so `/reset` does not revive the unsummarized prefix from SQLite unless we store the summary as the canonical transcript.

## Implementation sketch

- `src/processing/text/context/compact.py` - `should_compact`, `split_prefix_tail`, `apply_summary`.
- Call site: `build_messages_for_model` or the chat turn entry before it, so every backend shares it.
- Token estimate: chars/4 as v1. Replace with tiktoken / Anthropic count when energy already parses usage.
- Tests in `tests/test_context_compact.py` when implementation starts (user asked no new test scripts until then).

## Non-goals

- Compacting skills or memory pack.
- Recursive compact of the summary itself until the tail+summary still exceeds the ratio (one pass per trigger is enough. If still over, drop oldest summary and keep tail).
- Hermes `/compress` trajectory dumps for training.
- Codex internal guardian compact as a hidden subagent.

## References

- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Claude Code compaction (docs)](https://code.claude.com/docs/en/costs)
- [Hermes `/compress`](https://hermes-agent.nousresearch.com/docs/user-guide/cli)
- [DSH session log / deriveMessages](https://github.com/deepseek-ai/deepseek-harness/blob/main/docs/architecture.md)
