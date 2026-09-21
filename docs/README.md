# sophon

CLI and library wrappers around local Hugging Face weights, with optional retrieval
(LEANN index, LightRAG structure, Adaptive-RAG gate) and persistent SQLite chat memory.

Long-term product map (harness, editor, dashboard, subagents, swarms, self-evolution, constraints):
[VISION.md](VISION.md) and [direction/](direction/README.md). Subagent spawn and 3090/API placement: [direction/subagents.md](direction/subagents.md). Item backlog: [TODO.md](TODO.md).

## Install

From the repo root (either works):

```bash
uv sync
```

```powershell
cd C:\Software\Python\NLP\sophon
pip install -e .
```

Authenticate for gated Hub pulls when needed:

```bash
hf auth login
```

Optional: if `.env.example` ships in-tree, copy to `.env` and set `HF_TOKEN`. Override discovery with `SOPHON_ENV_FILE` or disable with `SOPHON_SKIP_DOTENV=1`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Package tree; setuptools `package-dir` maps **`sophon` -> `src`**. |
| `src/backend/` | HF runtime (`hf/backend`, `hf/paths`, `hf/const`, `hf/registry`) and Ollama (`ollama/backend`). |
| `src/cli/` | `infer`/`chat` backends (`inference.py`, `chat.py`), Click front-end (`terminal.py`), TTS flags (`flags/tts.py`), UI backends (`backends/`), desktop spawn (`host/`). |
| `src/eval/` | `sophon-benchmark` runner (`runner`, `scorers`, `task_spec`). |
| `src/processing/audio/` | Audio I/O pipelines (`speech/` TTS backends). |
| `src/processing/text/context/` | Assemble transient message lists for the model call. |
| `src/processing/text/memory/` | SQLite-backed chat memory (`SqliteMemoryStore`). |
| `src/processing/text/retrieval/` | Retrieval protocols and backends (noop, LEANN, LightRAG) plus Adaptive-RAG gate. |
| `src/utils/` | Hub downloads (`utils/download/`). |
| `src/utils/device/` | `.env` / project-root discovery (`env_bootstrap`), `sophon-system-check` (`system_check`). |
| `data/benchmarks/` | YAML task manifests (`data/benchmarks/README.md`). |
| `docs/trainer/` | LoRA SFT operator manual (models, hparams, datasets, stages, devices, adapters). |
| `config/finetune/` | Default and pipeline YAML recipes (`default.yaml`, `qa-chain.yaml`). |
| `docs/integrations/` | External host notes (for example Toad). |

Downloaded checkpoints usually live under `models/` (gitignored). Runtime chat logs and bundled UI assets live under `data/chat_logs/` and `data/assets/`.

Console entry points live in [`pyproject.toml`](../pyproject.toml): **`sophon-cli`**, **`sophon-infer`**, **`sophon-chat-cli`**, **`sophon-system-check`**, **`sophon-download-hf`**, **`sophon-hf-download`**, **`sophon-download-hf-debug`**, **`sophon-benchmark`**, **`sophon-rag-index`**, **`sophon-finetune`**.

Runtimes install **`sophon.processing.text.retrieval`**, **`sophon.processing.text.memory`**, **`sophon.processing.text.context`**, … as subpackages. For development without editable install, point `PYTHONPATH` at `src/` so shorthand imports (`import processing.text.retrieval`, `import cli.chat`) mirror the path bootstrap bundled with the scripts.

Optional retrieval extras:

```powershell
pip install -e ".[rag-leann]"
pip install -e ".[rag-lightrag]"
pip install -e ".[rag-full]"
```

```powershell
$env:PYTHONPATH = "C:\Software\Python\NLP\sophon\src"
```

## Environment

- `.env` discovery: `SOPHON_ENV_FILE`, then `./.env`, `./sophon/.env`, then walking
  up to the project root containing `pyproject.toml`. Disable with
  `SOPHON_SKIP_DOTENV=1`.
- Hugging Face: `HF_TOKEN` (required for gated repos), `SOPHON_HF_PRESET`,
  `SOPHON_HF_REVISION`, `GEMMA4_MODEL`, `GEMMA4_LOCAL_DIR`, `GEMMA4_REPO_ID`,
  `GEMMA4_REVISION`, `GEMMA4_QBIT`, `GEMMA4_QUANTIZATION`, `GEMMA4_SYSTEM_PROMPT`,
  `GEMMA4_THINKING`.
- Retrieval: `SOPHON_RAG`, `SOPHON_RAG_TOP_K`, `SOPHON_RAG_ADAPTIVE`,
  `SOPHON_RAG_STRUCTURE`, `SOPHON_LEANN_INDEX`, `SOPHON_LEANN_TOP_K`,
  `SOPHON_LEANN_ENTRYPOINT`, `SOPHON_LEANN_ROOT`, `SOPHON_LIGHTRAG_DIR`,
  `SOPHON_LIGHTRAG_MODE`, `SOPHON_LIGHTRAG_EMBED_MODEL`,
  `SOPHON_LIGHTRAG_EMBED_DIM`, `SOPHON_VAULT_PATH`.
- Memory: `SOPHON_MEMORY_DB`, `SOPHON_MEMORY_SESSION`, `SOPHON_MEMORY_USER`,
  `SOPHON_MEMORY_RECALL_TURNS`, `SOPHON_MEMORY_DIR`, `SOPHON_MEMORY_BUDGET_CHARS`,
  `SOPHON_MEMORY_TOOLS`. Five-tier layer: see
  [direction/memory.md](direction/memory.md).
- Skills: `SOPHON_SKILLS`, `SOPHON_SKILLS_DIRS`, `SOPHON_SKILLS_CATALOG_CHARS`,
  `SOPHON_SKILL_TOOLS`. Agent Skills catalog: see
  [direction/skills.md](direction/skills.md).
- Eval: `SOPHON_BENCH_DATA_DIR`, `SOPHON_BENCH_OUT_DIR`, `SOPHON_BENCH_BACKEND`,
  `SOPHON_BENCH_OLLAMA_MODEL`, `OLLAMA_HOST`.

## Click CLI (`sophon-cli`)

The interactive and one-shot helpers use [Click](https://click.palletsprojects.com/) instead of **`argparse`**.

- **`sophon-cli`** is the umbrella entry point: choose **`infer`** or **`chat`** as the first positional argument.
- **`sophon-infer`** and **`sophon-chat-cli`** stay as compatibility shims mapping to **`infer`** and **`chat`** with the legacy executable names preserved in **`--help`**.

```powershell
sophon-cli infer "Hello" --model .\models\my-model
sophon-cli chat --preset llama2_7b_chat
python -m sophon.cli infer --help
```


## CLI overview

| Command | Purpose |
| --- | --- |
| `sophon-cli` | Click umbrella for `infer` and `chat`. |
| `sophon-chat-cli` | Interactive REPL with optional RAG + memory. |
| `sophon-infer` | One-shot prompt -> reply. |
| `sophon-hf-download` | Wrap official `huggingface-cli` download by repo or preset. |
| `sophon-download-hf` | `snapshot_download` by preset, returns local path. |
| `sophon-download-hf-debug` | Verbose preset pull with tqdm + INFO logs. |
| `sophon-system-check` | Host/PyTorch/device probe, optional model-dir sizing. |
| `sophon-benchmark` | Run MMLU-style task manifests under `data/benchmarks`. |
| `sophon-rag-index` | Build default vault+project LEANN corpus (optional LightRAG insert). |

## `sophon-chat-cli`

Equivalent form: **`sophon-cli chat ...`**.

Interactive chat backed by a local HF model. Adds retrieval and SQLite memory when
the matching flags or env vars are set.

```powershell
$env:PYTHONPATH = "C:\Software\Python\NLP\sophon\src"
sophon-chat-cli --memory-db .\data\chat_logs\memory.sqlite --memory-session demo --rag leann --rag-index C:\path\to\my_index
```

### Model selection

- `--model PATH` - local directory containing `config.json`.
- `--preset KEY` - registry preset, resolves to `<root>/models/<slug>`.
  See `sophon-hf-download --preset` choices.
- `--quantization {none,4bit,8bit,lightweight}` - 4/8-bit requires CUDA + bitsandbytes.
- `--qbit {0,4,8}` - short form, overrides `--quantization`.

### Sampling and reasoning

- `--system TEXT` - default system prompt (`GEMMA4_SYSTEM_PROMPT` fallback).
- `--max-new-tokens N`
- `--thinking / --no-thinking` - opt into model-internal reasoning. Env: `GEMMA4_THINKING`.
- `--raw` - keep special tokens visible.
- `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--seed`.
- `--debug / --no-debug` - per-turn stats footer (also `SOPHON_CHAT_DEBUG`).

### Retrieval flags

Chat RAG is three layers: LEANN (dense index), LightRAG (graph structure), and
Adaptive-RAG (skip / single-hop / multi-hop before any retrieve). Memory stays
independent.

Default corpus is the Obsidian vault (`SOPHON_VAULT_PATH`) plus the project
tree, built into `data/rag/indexes/default`. When that index exists and
`SOPHON_RAG` is unset, chat auto-selects `--rag leann` with that path.
Pass `--rag noop` (or `SOPHON_RAG=noop`) to disable.

- `--rag {noop,leann,lightrag}` (auto: `leann` when default index exists, else `noop`).
  Env: `SOPHON_RAG`.
- `--rag-index PATH` - LEANN index path (default `data/rag/indexes/default` when present).
  Env: `SOPHON_LEANN_INDEX`.
- `--rag-top-k N` - default chunk count per query. Env: `SOPHON_RAG_TOP_K`.
- `--rag-adaptive / --no-rag-adaptive` - heuristic gate before retrieve
  (default on). Env: `SOPHON_RAG_ADAPTIVE` (`0`/`false`/`off` disables).
- `--rag-structure {none,lightrag}` - multi-hop / global queries.
  Env: `SOPHON_RAG_STRUCTURE`.
- `--rag-structure-dir PATH` - LightRAG working directory.
  Env: `SOPHON_LIGHTRAG_DIR`.

Build / rebuild the default corpus:

```powershell
$env:SOPHON_VAULT_PATH = "C:\Notes\Obsidian Notes"
sophon-rag-index --rebuild
# or: sophon-cli rag-index --rebuild
# or in chat: /rag-index --rebuild
```

LEANN-specific env still honored for the native backend: `SOPHON_LEANN_TOP_K`,
`SOPHON_LEANN_ENTRYPOINT` (custom callable as `module:attr`),
`SOPHON_LEANN_ROOT` (extra path prepended to `sys.path`).

LightRAG-specific env: `SOPHON_LIGHTRAG_MODE` (default `hybrid`),
`SOPHON_LIGHTRAG_EMBED_MODEL` (default `nomic-embed-text`),
`SOPHON_LIGHTRAG_EMBED_DIM` (default `768`). The backend expects Ollama LLM +
embed helpers from `lightrag-hku` unless you construct `LightRagNativeRetriever`
with custom callables.

Example (index + structure + adaptive):

```powershell
sophon-chat-cli --rag leann --rag-index C:\path\to\my_index `
  --rag-structure lightrag --rag-structure-dir C:\path\to\lightrag_workdir
```

`/rag-status` shows primary backend, structure backend, adaptive on/off, and the
last Adaptive-RAG decision (`skip` / `single_hop` / `multi_hop`).

### Eval (chat)

Impact checks live in chat (no separate Workshop pane):

- `/eval` - suite help + last summary
- `/eval model [task] [N]` - MMLU / Hellaswag via `eval.runner`
- `/eval rag [N]` - ablation: no-RAG vs default LEANN corpus
- `/eval train` - list indexed LoRA adapters (`data/training/adapters/index.json`)
- `/finetune` - LoRA SFT on a local HF causal preset. YAML in `config/finetune/`. Operator notes: [trainer/](trainer/README.md).
- CLI: `sophon-benchmark --task rag_ablation --limit 20 --preset <preset>`

Artifacts land under `data/exps/<run_id>/`.

### Memory flags

- `--memory-db PATH` - SQLite file or `:memory:`. Default on (`data/memory/memory.db`). Env: `SOPHON_MEMORY_DB`. Set `0` to disable.
- `--memory-session ID` - reuse a session id to recall its prior turns. Default is
  `session_<timestamp>`. Env: `SOPHON_MEMORY_SESSION`.
- `--memory-user ID` - optional user id stored next to each message. Env:
  `SOPHON_MEMORY_USER`.
- `--memory-recall-turns N` - trailing turns injected as context. Env:
  `SOPHON_MEMORY_RECALL_TURNS`.

Chat runs the five-tier memory layer by default (working, episodic, semantic, procedural, retrieved). Durable facts live in
`.sophon/memory/facts.md` and are promoted from scratch through the editor Review
tab. Model tools (`memory_view` / `memory_search` / `memory_read` / `memory_write` /
`memory_propose`) are gated by `SOPHON_MEMORY_TOOLS`. The model may write scratch
and queue a Review ask. It cannot write `facts.md`. Human
commands: `/memory status|list|search|note|promote|revoke|compact`. Full contract:
[direction/memory.md](direction/memory.md).

Skills are on by default. Chat scans `.sophon/skills`, `~/.sophon/skills`, and
read-only `.cursor/skills` / `.claude/skills` (not `skills-cursor`). The catalog
is injected each turn. Full `SKILL.md` bodies attach via `/skill attach` or
`skill_read`. Create and import queue Review. Model tools
(`skill_list` / `skill_read` / `skill_read_file`) are gated by
`SOPHON_SKILL_TOOLS`. Human commands:
`/skill list|show|attach|detach|create|import` (`/skills` lists). Full contract:
[direction/skills.md](direction/skills.md).

The chat loop persists each successful `(user, assistant)` pair to SQLite. Failed
generations are not persisted. `/regen` re-runs without duplicating the user row.

### Slash commands

`/help` prints grouped sections: session, model, generation, tools, speech,
memory, skills, eval, train. `/clear` empties the visible transcript and starts
a new session id (facts and scratch stay). `/reset` clears model messages and
the on-screen log but keeps the same session. `/help generation` shows one section. Shortcuts: `?` help,
`q` quit, `s` save, `u` pop, `r` regen, `d` debug.

Triple-quoted input (`"""multi-line"""`) is gathered until the closing fence.

## `sophon-infer`

Equivalent form: **`sophon-cli infer PROMPT`** (also **`python -m sophon.cli infer PROMPT`**).

One-shot prompt -> reply. Same model/quantization/sampling flags as the chat CLI, except **`sophon-chat-cli`** alone accepts **`--preset`**.

```powershell
sophon-infer "Write a haiku about sophon." --model .\models\meta-llama-Llama-2-7b-chat-hf --quantization 4bit
```

Use `--system`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`,
`--seed`, `--max-new-tokens`, `--thinking/--no-thinking`, `--raw`.

Checkpoint directory resolution (`sophon-infer`, benchmarks, chat model args):

1. `--model PATH` when passed.
2. Else `GEMMA4_MODEL`.
3. Else `GEMMA4_LOCAL_DIR` or preset layout / registry default (`models/...`).
   The chosen directory **must contain `config.json`**; inference does not pull Hub ids implicitly.

`load_processor_and_model` (`src/backend/hf/backend.py`) enables Transformers tqdm progress for weight shards (writes to stderr). If Hub suppressed bars globally, the loader enables them during that call only.

| Quantization flag | Behavior |
| --- | --- |
| `none` | Full weights. On Apple Silicon with MPS, the HF path prefers MPS placement when available; on CUDA hosts, `device_map="auto"` is typical. |
| `4bit`, `8bit`, `lightweight` | Bitsandbytes path; **CUDA** in typical setups on this project. Avoid on Apple Silicon. |

Default when **`GEMMA4_QUANTIZATION`** is unset: **`none`** if MPS is usable; **`4bit`** when **`torch.cuda.is_available()`**; otherwise **`none`** (CPU-only PyTorch). **`PYTORCH_ENABLE_MPS_FALLBACK`** defaults to enable mixed-kernel fallback on Metal.

Thinking mode CLI: **`--thinking` / `--no-thinking`**. If omitted, **`GEMMA4_THINKING`** (`1` / `true` / `yes`) enables it.

Accelerate **`device_map="auto"`**: on CUDA/CPU-ish hosts without MPS-only placement, HF + Accelerate may spread layers automatically. **`device_map="auto"` alone does not make bitsandbytes-quantized checkpoints run on MPS**; quantized loads stay CUDA-facing here.

## Model downloads

Three entry points exist for different cases.

### `sophon-hf-download`

Shell-style wrapper around `python -m huggingface_hub.cli.hf download`.

```powershell
sophon-hf-download meta-llama/Llama-2-7b-chat-hf --local-dir .\models\meta-llama-Llama-2-7b-chat-hf
sophon-hf-download --preset llama2_7b_chat
sophon-hf-download                          # falls back to SOPHON_HF_PRESET / registry default
```

Args: `repo_id` (positional, optional), `--preset KEY`, `--local-dir PATH`,
`--revision REF`, `--project-root PATH`. Honors `HF_TOKEN`,
`SOPHON_HF_REVISION`.

### `sophon-download-hf`

Library-style `snapshot_download` with a default preset.

```powershell
sophon-download-hf --preset llama2_7b_chat --verbose
sophon-download-hf --list-presets
sophon-download-hf --repo-id google/gemma-4-E2B-it --local-dir .\models\gemma-4-E2B-it
```

Args: `--preset`, `--repo-id`, `--local-dir`, `--revision`, `--list-presets`,
`--verbose`.

Defaults when omitting positional repo arguments on `sophon-download-hf`: preset resolves to **`llama2_7b_chat`** unless overridden. Destination defaults to registry layout under `models/...`. **`gemma4_31b_it`** paths accept `GEMMA4_REPO_ID`, `GEMMA4_LOCAL_DIR`, and `GEMMA4_REVISION` when matching flags are absent.

### `sophon-download-hf-debug`

Verbose pull with INFO logs and tqdm; default preset is `llama2_7b_chat` (override
via `--preset` or `SOPHON_DEBUG_PRESET`). Useful when diagnosing gated-repo errors.

## `sophon-system-check`

```powershell
sophon-system-check                       # describe all devices
sophon-system-check --device auto         # one-line recommendation
sophon-system-check --json                # machine-readable report
sophon-system-check --allocate-mib 256    # smoke-test allocation
```

Args: `--device {auto,cuda,mps,cpu,all}`, `--cuda-device N`, `--allocate-mib N`,
`--probe-device SPEC`, `--model PATH`, `--json`.

## `sophon-benchmark`

Run task manifests stored under `data/benchmarks` (or `SOPHON_BENCH_DATA_DIR`).

```powershell
sophon-benchmark --list
sophon-benchmark --task mmlu --limit 50 --preset llama2_7b_chat
sophon-benchmark --task all --backend ollama --ollama-model llama3:8b-instruct
```

Args (selection): `--task ID|all`, `--data-dir`, `--list`,
`--backend {hf,ollama}`, `--model`, `--quantization`, `--qbit`, `--thinking`,
`--ollama-model`, `--ollama-host`, `--limit`, `--split`, `--max-new-tokens`,
`--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--seed`,
`--out-dir`, `--run-id`, `--run-label`.

Run artifacts (default `<repo>/data/exps/` or `SOPHON_BENCH_OUT_DIR`), layout `<out>/<run_id>/<task>/`:

- **`experiment_summary.json`** (under `<run_id>/`): fingerprint digest plus pointers to each task `summary.json` / `predictions.jsonl`.
- **`predictions.jsonl`** (per task): one JSON object per example (raw response text, routed answer extraction, labels, timings).
- **`summary.json`** (per task): aggregate accuracy and timing plus serialized `RunConfig`.

Auto **`run_id`** is UTC timestamp (second + microsecond) + backend + slug(model dir name + path digest on HF, or Ollama model name) + quantization + thinking flag + 12-char SHA256 digest of CLI knobs (temperature, seeds, split, limit, …). Override with **`--run-id`**, append a note with **`--run-label`** (`__suffix`).

## Retrieval module

`sophon.processing.text.retrieval` exposes the protocols, factory, and bundled backends.

```python
from sophon.processing.text.retrieval import load_rag_retriever, RetrievalQuery

retriever = load_rag_retriever("leann", native_index_path=r"C:\path\to\my_index")
hits = retriever.retrieve(RetrievalQuery("what is X?", params={"top_k": 8}))
for chunk in hits.chunks:
    print(chunk["id"], chunk["score"], chunk["text"][:120])
```

Bundled backends:

- `noop` - returns an empty `RetrievalResult`.
- `leann` (native) - thin wrapper around `leann.LeannSearcher`, requires the
  `leann` distribution. Set `SOPHON_LEANN_INDEX` or pass `native_index_path=`.
- `lightrag` (native) - wrapper around HKUDS LightRAG (`lightrag-hku`). Set
  `SOPHON_LIGHTRAG_DIR` or pass `structure_dir=`. Used from chat as
  `--rag-structure lightrag` for multi-hop routing.
- Callable adapter - `load_rag_retriever("noop", entrypoint="pkg.mod:fn")` wraps
  any `(RetrievalQuery) -> RetrievalResult | mapping`.

Adaptive-RAG (`processing.text.retrieval.adaptive.decide_retrieval`) is a
deterministic text policy (not a third index): phatic / empty queries skip,
comparison / cross-document cues prefer `multi_hop` when a structure retriever
is configured, otherwise fall back to `single_hop` (LEANN).

Query performance estimates (latency, hit count, top/mean score, score margin):
each retrieve attaches `extras["metrics"]`. Chat also stores
`extras["adaptive"]` with the gate decision. In chat, `/rag-status` shows the last
query plus a rolling probe window. `/rag-probe [N] [query]` repeats search and
prints mean/p50/p95 latency and empty-hit rate. These are cost/confidence proxies,
not labeled recall (hit@k needs a JSONL of `{query, relevant_ids}`).

Default corpus indexing: `sophon-rag-index` / `/rag-index` walks
`SOPHON_VAULT_PATH` plus the project tree into `data/rag/indexes/default`
via `LeannBuilder`. Optional LightRAG insert uses `SOPHON_LIGHTRAG_DIR`.
You can still build indexes with LEANN's own CLI for custom corpora.

## Memory module

`sophon.processing.text.memory` provides `SqliteMemoryStore` with schema v1
(`sessions`, `messages`, `schema_meta`).

```python
from sophon.processing.text.memory import SqliteMemoryStore, MemoryScope, open_memory_store

store = open_memory_store(r".\data\chat_logs\memory.sqlite")  # honors SOPHON_MEMORY_DB
scope = MemoryScope(session_id="demo", user_id="philipp")
store.append_turn(scope, "user", "hello")
store.append_turn(scope, "assistant", "hi there", meta={"backend": "hf"})
for turn in store.load_recent_turns(scope, limit=6):
    print(turn.role, turn.content)
store.close()
```

Persistence is opt-in. Without a path, the chat loop runs ephemeral.

## Context module

`sophon.processing.text.context.build_messages_for_model` composes the message list sent to the
model. It does not mutate the canonical transcript; the chat CLI keeps
`state.messages` clean and assembles a transient list per turn (design fork A).

```python
from sophon.processing.text.context import build_messages_for_model

built = build_messages_for_model(
    base_messages=state.messages,
    retrieval=retrieval_result,
    memory_turns=memory_store.load_recent_turns(scope, 6),
    system_text="You are concise.",
    max_retrieval_chunks=5,
    max_retrieval_chars=800,
    max_memory_turns=6,
)
result = generate_response(processor, model, built.messages, ...)
```

`ContextBuildResult` reports which blocks were injected for debugging
(`/debug on` in the chat CLI prints them).

## Limitations

- **Unified memory hosts**: Dense `none` checkpoints (for example Gemma 4 31B) need unified memory proportional to checkpoint size plus runtime overhead. Prefer quantization or smaller presets when constrained.
- **Multimodal** Gemma checkpoints require multimodal preprocessing and model classes distinct from these text-centric CLIs. Treat current chat/infer helpers as **text-first** wrappers.
- **MPS warmup**: Accelerate-backed loads may attempt large allocator warm-ups; recent Transformers shims clamp or bypass some paths when targeting Metal. Failures manifest as oversized allocation errors (`Invalid buffer size`); switch to quantized CPU/CUDA snapshots or resize hardware expectations.

## Quick recipes

End-to-end RAG chat session:

```powershell
$env:PYTHONPATH = "C:\Software\Python\NLP\sophon\src"
$env:HF_TOKEN = "<token>"
sophon-chat-cli `
  --preset llama2_7b_chat `
  --quantization 4bit `
  --memory-db .\data\chat_logs\memory.sqlite `
  --memory-session demo `
  --rag leann `
  --rag-index .\indexes\my_corpus
```

Reuse the session later:

```powershell
sophon-chat-cli --memory-db .\data\chat_logs\memory.sqlite --memory-session demo
```

Inspect what is loaded inside the chat:

```
/rag-status
/rag-probe 10 what is sophon retrieval
/memory-status
/debug on
```

Wipe stored memory for the active session:

```
/memory-clear
```
