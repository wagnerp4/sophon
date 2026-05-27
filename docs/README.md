# mithril

CLI and library wrappers around local Hugging Face weights, with optional retrieval
(LEANN) and persistent SQLite chat memory.

## Install

From the repo root (either works):

```bash
uv sync
```

```powershell
cd C:\Software\Python\NLP\mithril
pip install -e .
```

Authenticate for gated Hub pulls when needed:

```bash
hf auth login
```

Optional: if `.env.example` ships in-tree, copy to `.env` and set `HF_TOKEN`. Override discovery with `MITHRIL_ENV_FILE` or disable with `MITHRIL_SKIP_DOTENV=1`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Package tree; setuptools `package-dir` maps **`mithril` -> `src`**. |
| `src/backend/` | HF runtime (`hf/backend`, `hf/paths`, `hf/const`, `hf/registry`) and Ollama (`ollama/backend`). |
| `src/cli/` | `infer`/`chat` backends (`inference.py`, `chat.py`), Click front-end (`terminal.py`), shared parsing (`terminal_setup.py`). |
| `src/eval/` | `mithril-benchmark` runner (`runner`, `scorers`, `task_spec`). |
| `src/processing/audio/` | Audio I/O pipelines (`speech/` TTS backends). |
| `src/processing/text/context/` | Assemble transient message lists for the model call. |
| `src/processing/text/memory/` | SQLite-backed chat memory (`SqliteMemoryStore`). |
| `src/processing/text/retrieval/` | Retrieval protocols and backends (noop, LEANN, callable). |
| `src/utils/` | Hub downloads (`utils/download/`). |
| `src/utils/device/` | `.env` / project-root discovery (`env_bootstrap`), `mithril-system-check` (`system_check`). |
| `data/benchmarks/` | YAML task manifests (`data/benchmarks/README.md`). |
| `docs/integrations/` | External host notes (for example Toad). |

Downloaded checkpoints usually live under `models/` (gitignored). Runtime chat logs and bundled UI assets live under `data/chat_logs/` and `data/assets/`.

Console entry points live in [`pyproject.toml`](../pyproject.toml): **`mithril-cli`**, **`mithril-infer`**, **`mithril-chat-cli`**, **`mithril-system-check`**, **`mithril-download-hf`**, **`mithril-hf-download`**, **`mithril-download-hf-debug`**, **`mithril-benchmark`**.

Runtimes install **`mithril.processing.text.retrieval`**, **`mithril.processing.text.memory`**, **`mithril.processing.text.context`**, … as subpackages. For development without editable install, point `PYTHONPATH` at `src/` so shorthand imports (`import processing.text.retrieval`, `import cli.chat`) mirror the path bootstrap bundled with the scripts.

Optional retrieval extras:

```powershell
pip install -e ".[rag-leann]"
```

```powershell
$env:PYTHONPATH = "C:\Software\Python\NLP\mithril\src"
```

## Environment

- `.env` discovery: `MITHRIL_ENV_FILE`, then `./.env`, `./mithril/.env`, then walking
  up to the project root containing `pyproject.toml`. Disable with
  `MITHRIL_SKIP_DOTENV=1`.
- Hugging Face: `HF_TOKEN` (required for gated repos), `MITHRIL_HF_PRESET`,
  `MITHRIL_HF_REVISION`, `GEMMA4_MODEL`, `GEMMA4_LOCAL_DIR`, `GEMMA4_REPO_ID`,
  `GEMMA4_REVISION`, `GEMMA4_QBIT`, `GEMMA4_QUANTIZATION`, `GEMMA4_SYSTEM_PROMPT`,
  `GEMMA4_THINKING`.
- Retrieval: `MITHRIL_RAG`, `MITHRIL_RAG_TOP_K`, `MITHRIL_LEANN_INDEX`,
  `MITHRIL_LEANN_TOP_K`, `MITHRIL_LEANN_ENTRYPOINT`, `MITHRIL_LEANN_ROOT`.
- Memory: `MITHRIL_MEMORY_DB`, `MITHRIL_MEMORY_SESSION`, `MITHRIL_MEMORY_USER`,
  `MITHRIL_MEMORY_RECALL_TURNS`.
- Eval: `MITHRIL_BENCH_DATA_DIR`, `MITHRIL_BENCH_OUT_DIR`, `MITHRIL_BENCH_BACKEND`,
  `MITHRIL_BENCH_OLLAMA_MODEL`, `OLLAMA_HOST`.

## Click CLI (`mithril-cli`)

The interactive and one-shot helpers use [Click](https://click.palletsprojects.com/) instead of **`argparse`**.

- **`mithril-cli`** is the umbrella entry point: choose **`infer`** or **`chat`** as the first positional argument.
- **`mithril-infer`** and **`mithril-chat-cli`** stay as compatibility shims mapping to **`infer`** and **`chat`** with the legacy executable names preserved in **`--help`**.

```powershell
mithril-cli infer "Hello" --model .\models\my-model
mithril-cli chat --preset llama2_7b_chat
python -m mithril.cli infer --help
```


## CLI overview

| Command | Purpose |
| --- | --- |
| `mithril-cli` | Click umbrella for `infer` and `chat`. |
| `mithril-chat-cli` | Interactive REPL with optional RAG + memory. |
| `mithril-infer` | One-shot prompt -> reply. |
| `mithril-hf-download` | Wrap official `huggingface-cli` download by repo or preset. |
| `mithril-download-hf` | `snapshot_download` by preset, returns local path. |
| `mithril-download-hf-debug` | Verbose preset pull with tqdm + INFO logs. |
| `mithril-system-check` | Host/PyTorch/device probe, optional model-dir sizing. |
| `mithril-benchmark` | Run MMLU-style task manifests under `data/benchmarks`. |

## `mithril-chat-cli`

Equivalent form: **`mithril-cli chat ...`**.

Interactive chat backed by a local HF model. Adds retrieval and SQLite memory when
the matching flags or env vars are set.

```powershell
$env:PYTHONPATH = "C:\Software\Python\NLP\mithril\src"
mithril-chat-cli --memory-db .\data\chat_logs\memory.sqlite --memory-session demo --rag leann --rag-index C:\path\to\my_index
```

### Model selection

- `--model PATH` - local directory containing `config.json`.
- `--preset KEY` - registry preset, resolves to `<root>/models/<slug>`.
  See `mithril-hf-download --preset` choices.
- `--quantization {none,4bit,8bit,lightweight}` - 4/8-bit requires CUDA + bitsandbytes.
- `--qbit {0,4,8}` - short form, overrides `--quantization`.

### Sampling and reasoning

- `--system TEXT` - default system prompt (`GEMMA4_SYSTEM_PROMPT` fallback).
- `--max-new-tokens N`
- `--thinking / --no-thinking` - opt into model-internal reasoning. Env: `GEMMA4_THINKING`.
- `--raw` - keep special tokens visible.
- `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--seed`.
- `--debug / --no-debug` - per-turn stats footer (also `MITHRIL_CHAT_DEBUG`).

### Retrieval flags

- `--rag {noop,leann}` (default `noop`). Env: `MITHRIL_RAG`.
- `--rag-index PATH` - LEANN index basename (sibling `*.meta.json`).
  Env: `MITHRIL_LEANN_INDEX`.
- `--rag-top-k N` - default chunk count per query. Env: `MITHRIL_RAG_TOP_K`.

LEANN-specific env still honored for the native backend: `MITHRIL_LEANN_TOP_K`,
`MITHRIL_LEANN_ENTRYPOINT` (custom callable as `module:attr`),
`MITHRIL_LEANN_ROOT` (extra path prepended to `sys.path`).

### Memory flags

- `--memory-db PATH` - SQLite file or `:memory:`. Env: `MITHRIL_MEMORY_DB`.
- `--memory-session ID` - reuse a session id to recall its prior turns. Default is
  `session_<timestamp>`. Env: `MITHRIL_MEMORY_SESSION`.
- `--memory-user ID` - optional user id stored next to each message. Env:
  `MITHRIL_MEMORY_USER`.
- `--memory-recall-turns N` - trailing turns injected as context. Env:
  `MITHRIL_MEMORY_RECALL_TURNS`.

The chat loop persists each successful `(user, assistant)` pair to SQLite. Failed
generations are not persisted. `/regen` re-runs without duplicating the user row.

### Slash commands

`/help` (or `?`), `/quit` (`q`), `/reset`, `/save [PATH]` (`s`),
`/pop` (`u`), `/regen` (`r`), `/debug [on|off]` (`d`), `/stats [reset]`,
`/tokens`, `/system [show|clear|TEXT]`, `/temp`, `/top-p`, `/top-k`, `/max`,
`/seed`, `/rep`, `/rag-status`, `/memory-status`, `/memory-clear`.

Triple-quoted input (`"""multi-line"""`) is gathered until the closing fence.

## `mithril-infer`

Equivalent form: **`mithril-cli infer PROMPT`** (also **`python -m mithril.cli infer PROMPT`**).

One-shot prompt -> reply. Same model/quantization/sampling flags as the chat CLI, except **`mithril-chat-cli`** alone accepts **`--preset`**.

```powershell
mithril-infer "Write a haiku about mithril." --model .\models\meta-llama-Llama-2-7b-chat-hf --quantization 4bit
```

Use `--system`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`,
`--seed`, `--max-new-tokens`, `--thinking/--no-thinking`, `--raw`.

Checkpoint directory resolution (`mithril-infer`, benchmarks, chat model args):

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

### `mithril-hf-download`

Shell-style wrapper around `python -m huggingface_hub.cli.hf download`.

```powershell
mithril-hf-download meta-llama/Llama-2-7b-chat-hf --local-dir .\models\meta-llama-Llama-2-7b-chat-hf
mithril-hf-download --preset llama2_7b_chat
mithril-hf-download                          # falls back to MITHRIL_HF_PRESET / registry default
```

Args: `repo_id` (positional, optional), `--preset KEY`, `--local-dir PATH`,
`--revision REF`, `--project-root PATH`. Honors `HF_TOKEN`,
`MITHRIL_HF_REVISION`.

### `mithril-download-hf`

Library-style `snapshot_download` with a default preset.

```powershell
mithril-download-hf --preset llama2_7b_chat --verbose
mithril-download-hf --list-presets
mithril-download-hf --repo-id google/gemma-4-E2B-it --local-dir .\models\gemma-4-E2B-it
```

Args: `--preset`, `--repo-id`, `--local-dir`, `--revision`, `--list-presets`,
`--verbose`.

Defaults when omitting positional repo arguments on `mithril-download-hf`: preset resolves to **`llama2_7b_chat`** unless overridden. Destination defaults to registry layout under `models/...`. **`gemma4_31b_it`** paths accept `GEMMA4_REPO_ID`, `GEMMA4_LOCAL_DIR`, and `GEMMA4_REVISION` when matching flags are absent.

### `mithril-download-hf-debug`

Verbose pull with INFO logs and tqdm; default preset is `llama2_7b_chat` (override
via `--preset` or `MITHRIL_DEBUG_PRESET`). Useful when diagnosing gated-repo errors.

## `mithril-system-check`

```powershell
mithril-system-check                       # describe all devices
mithril-system-check --device auto         # one-line recommendation
mithril-system-check --json                # machine-readable report
mithril-system-check --allocate-mib 256    # smoke-test allocation
```

Args: `--device {auto,cuda,mps,cpu,all}`, `--cuda-device N`, `--allocate-mib N`,
`--probe-device SPEC`, `--model PATH`, `--json`.

## `mithril-benchmark`

Run task manifests stored under `data/benchmarks` (or `MITHRIL_BENCH_DATA_DIR`).

```powershell
mithril-benchmark --list
mithril-benchmark --task mmlu --limit 50 --preset llama2_7b_chat
mithril-benchmark --task all --backend ollama --ollama-model llama3:8b-instruct
```

Args (selection): `--task ID|all`, `--data-dir`, `--list`,
`--backend {hf,ollama}`, `--model`, `--quantization`, `--qbit`, `--thinking`,
`--ollama-model`, `--ollama-host`, `--limit`, `--split`, `--max-new-tokens`,
`--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--seed`,
`--out-dir`, `--run-id`, `--run-label`.

Run artifacts (default `<repo>/data/exps/` or `MITHRIL_BENCH_OUT_DIR`), layout `<out>/<run_id>/<task>/`:

- **`experiment_summary.json`** (under `<run_id>/`): fingerprint digest plus pointers to each task `summary.json` / `predictions.jsonl`.
- **`predictions.jsonl`** (per task): one JSON object per example (raw response text, routed answer extraction, labels, timings).
- **`summary.json`** (per task): aggregate accuracy and timing plus serialized `RunConfig`.

Auto **`run_id`** is UTC timestamp (second + microsecond) + backend + slug(model dir name + path digest on HF, or Ollama model name) + quantization + thinking flag + 12-char SHA256 digest of CLI knobs (temperature, seeds, split, limit, …). Override with **`--run-id`**, append a note with **`--run-label`** (`__suffix`).

## Retrieval module

`mithril.processing.text.retrieval` exposes the protocols, factory, and bundled backends.

```python
from mithril.processing.text.retrieval import load_rag_retriever, RetrievalQuery

retriever = load_rag_retriever("leann", native_index_path=r"C:\path\to\my_index")
hits = retriever.retrieve(RetrievalQuery("what is X?", params={"top_k": 8}))
for chunk in hits.chunks:
    print(chunk["id"], chunk["score"], chunk["text"][:120])
```

Bundled backends:

- `noop` - returns an empty `RetrievalResult`.
- `leann` (native) - thin wrapper around `leann.LeannSearcher`, requires the
  `leann` distribution. Set `MITHRIL_LEANN_INDEX` or pass `native_index_path=`.
- Callable adapter - `load_rag_retriever("noop", entrypoint="pkg.mod:fn")` wraps
  any `(RetrievalQuery) -> RetrievalResult | mapping`.

Indexing (building corpora) is intentionally out of scope. Use LEANN's own tools
(`LeannBuilder`, `leann` CLI) to produce indexes consumed here.

## Memory module

`mithril.processing.text.memory` provides `SqliteMemoryStore` with schema v1
(`sessions`, `messages`, `schema_meta`).

```python
from mithril.processing.text.memory import SqliteMemoryStore, MemoryScope, open_memory_store

store = open_memory_store(r".\data\chat_logs\memory.sqlite")  # honors MITHRIL_MEMORY_DB
scope = MemoryScope(session_id="demo", user_id="philipp")
store.append_turn(scope, "user", "hello")
store.append_turn(scope, "assistant", "hi there", meta={"backend": "hf"})
for turn in store.load_recent_turns(scope, limit=6):
    print(turn.role, turn.content)
store.close()
```

Persistence is opt-in. Without a path, the chat loop runs ephemeral.

## Context module

`mithril.processing.text.context.build_messages_for_model` composes the message list sent to the
model. It does not mutate the canonical transcript; the chat CLI keeps
`state.messages` clean and assembles a transient list per turn (design fork A).

```python
from mithril.processing.text.context import build_messages_for_model

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
$env:PYTHONPATH = "C:\Software\Python\NLP\mithril\src"
$env:HF_TOKEN = "<token>"
mithril-chat-cli `
  --preset llama2_7b_chat `
  --quantization 4bit `
  --memory-db .\data\chat_logs\memory.sqlite `
  --memory-session demo `
  --rag leann `
  --rag-index .\indexes\my_corpus
```

Reuse the session later:

```powershell
mithril-chat-cli --memory-db .\data\chat_logs\memory.sqlite --memory-session demo
```

Inspect what is loaded inside the chat:

```
/rag-status
/memory-status
/debug on
```

Wipe stored memory for the active session:

```
/memory-clear
```
