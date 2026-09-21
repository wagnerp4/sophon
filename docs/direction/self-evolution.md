# Self-evolution (Lembas and memory)

## Role

Self-evolution means the system **keeps procedures that measured well** and **drops ones that did not**. It does not mean the running process rewrites its own Python.

Working name for skills: **Lembas**. On disk they are Agent Skills folders under `.sophon/skills` (see [skills.md](skills.md)). Memory is a five-tier layer (`SOPHON_MEMORY_*`, see [memory.md](memory.md)). Retrieval remains LEANN / LightRAG / Adaptive-RAG.

## Current behavior

- Memory: five tiers (working, episodic, semantic, procedural, retrieved). Model writes scratch only. Durable facts are promoted by the human through Review. See [memory.md](memory.md).
- RAG: rebuild via `sophon-rag-index` / `/rag-index`. Vault + project default corpus.
- Eval in Chat: `/eval model` (MMLU, Hellaswag), `/eval rag` (ablation vs default index), `/eval train` (stub: list run dirs).
- Finetune CLI exists (`sophon-finetune`, PEFT / optional Unsloth). Not wired to "evolve from downvoted turns".
- Feedback (flag reasoning errors, upvote/downvote turns) is backlog only.

## What may evolve

| Artifact | How it may change | Gate |
| --- | --- | --- |
| Lembas skill | Prompt, tool allowlist, few-shots, tests | Eval suite + human promote |
| Memory: working (scratch) | Model or human notes for this session | Deny-list redaction, low trust |
| Memory: episodic | Session summaries appended on close/compact | System write, medium trust |
| Memory: semantic (facts) | Durable facts in `facts.md` | Promote from scratch via Review, dedup, no secrets |
| Memory: procedural | Pointers to Lembas skills in `procedures.md` | Human edit |
| RAG index | Rebuilt from vault/project | Build job, not a silent background rewrite of queries |
| Adapters | LoRA from traces / datasets | Isolated dir, `/eval train` after, never overwrite the base preset in place |
| Dashboard tile set | `.env` only | Human edit |

## What must not evolve unsupervised

- `src/` harness and TUI code
- `pyproject.toml` and lockfile
- OAuth client secrets, `.env`, Google token stores
- Git remotes, Overleaf tokens
- Harness policy that expands permissions

## Constraints

- **Measure first.** A skill that cannot be eval'd stays experimental (`/eval` or a skill-level fixture).
- **Version skills.** `lembas/<id>@<rev>`. Rollback is a file copy, not a model hallucinating the previous prompt.
- **Human promote.** Chat may *propose* a skill from a trace. Write to the skills tree goes through Review, same as editor diffs.
- **No secret leakage into skills.** Redact tokens when mining traces.
- **Feedback is labels, not truth.** Downvotes train a dataset. They do not immediately change system prompt.
- **Adaptive-RAG already skips retrieve.** Do not add a second silent "maybe skip" layer without eval.

## Future direction

1. Skill format is the public Agent Skills `SKILL.md` under `.sophon/skills` (and scanned Cursor/Claude trees). Versioning `lembas/<id>@<rev>` and eval-before-promote stay follow-ups. Attach by `/skill attach` or `skill_read`.
2. Mine skills from accepted editor batches and high-vote turns. Require `/eval` on a held-out set of similar tasks.
3. Self-evolving memory: periodic summarize + conflict detect. Keep `SOPHON_MEMORY_RECALL_TURNS` as a cap.
4. Finetune on error patterns: export traces with flags, `sophon-finetune`, then A/B in `/eval model`. Recipe and isolation: [trainer/](../trainer/README.md). Do not auto-load the adapter into the default TUI preset.
5. Structure pack (`structure/`: rules, style, form, few-shots) is input to CodeAgent, versioned like skills.
6. Crossref / semantic search for ResearchAgent stay tools with eval (citation hallucination rate), not memory writes.

## Non-goals

- Recursive self-modification of the interpreter.
- Promoting a skill because the model said it worked.
- Fine-tuning on the full chat DB without a filter (PII, secrets, failed tools).
