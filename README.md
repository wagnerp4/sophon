<div align="center">

# lmwrap

[![Python](https://img.shields.io/badge/python-3.10+-7f9a97?style=flat-square&logo=python&logoColor=white&labelColor=444444)](./pyproject.toml)
[![Model](https://img.shields.io/badge/model-Gemma%204%2031B-000000?style=flat-square&logo=huggingface&logoColor=white&labelColor=444444)](https://huggingface.co/google/gemma-4-31B-it)
[![uv](https://img.shields.io/badge/tooling-uv-7f9a97?style=flat-square&labelColor=444444)](https://docs.astral.sh/uv/)
[![Transformers](https://img.shields.io/badge/stack-Transformers-000000?style=flat-square&logo=huggingface&logoColor=white&labelColor=444444)](https://github.com/huggingface/transformers)

Small utilities for **[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)** locally using [uv](https://docs.astral.sh/uv/). Hub download, inference CLI, and terminal chat CLI. All inference paths assume **weights are already on disk** (no runtime Hub fetch for the model).

</div>

## Requirements

- **Python** 3.10 or newer (see `pyproject.toml`).
- **Pillow** (`pillow` in `pyproject.toml`). Transformers registers it for `Gemma4Processor` even when you only use text.
- **Torchvision** (`torchvision` in `pyproject.toml`). The Gemma 4 processor stack includes a video preprocessor that imports torchvision.
- A **Hugging Face account** with access to the gated Gemma checkpoint. Accept the terms on the model page and authenticate (see below).
- **Disk space** for the full checkpoint (tens of GB depending on shards and revision).
- For **CUDA** PCs, optional **4-bit / 8-bit** loading uses [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (typically NVIDIA).
- **Apple Silicon (macOS)**: use **quantization `none`** and the **Metal (MPS)** path (see device section). Full-precision 31B fits only on machines with enough unified memory.

## Setup

Clone the repo, then install dependencies:

```bash
uv sync
```

Optional: keep secrets in a **local** `.env` file (ignored by git). Copy `.env.example` to `.env` next to `pyproject.toml` and set `HF_TOKEN`. Override path with `LMWRAP_ENV_FILE` or disable with `LMWRAP_SKIP_DOTENV=1`.

Optional: authenticate for Hub downloads (`HF_TOKEN`, or):

```bash
hf auth login
```

## Repository layout


| Path              | Purpose                                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `src/`            | Package files live here; `**lmwrap**` maps onto `src/` via setuptools `**package-dir**` (`lmwrap.app` → `src/app`, etc.). |
| `src/__init__.py` | Package root for `**lmwrap**`.                                                                                            |
| `src/app/`        | CLI inference (`inference`), interactive chat CLI (`chat_cli`), system check, placeholder `main`. |
| `src/backend/`    | Hugging Face Gemma load path (`gemma_backend`, `gemma_paths`) and Ollama HTTP client (`ollama_backend`).                  |
| `src/utils/`      | Hub download CLI (`download_hf_model`) and preset registry (`registry`).                                                  |


Console scripts from `pyproject.toml`: `**lmwrap-infer**`, `**lmwrap-chat-cli**`, `**lmwrap-system-check**`, `**lmwrap-download-hf**`, `**lmwrap-hf-download**`, and `**lmwrap-download-hf-debug**`.

Downloaded weights are usually kept under `models/` (that directory is listed in `.gitignore` so large files are not committed).

## Downloading weights

Thin wrapper around the Hub library (`snapshot_download`) and a separate entry point that shells to the official `**hf download**` CLI (same as `python -m huggingface_hub.cli.hf download`):

```bash
uv run lmwrap-hf-download --preset llama2_7b_chat
uv run lmwrap-hf-download meta-llama/Llama-2-7b-chat-hf --local-dir models/meta-llama-Llama-2-7b-chat-hf
```

The `**lmwrap-download-hf**` script mirrors the Hub repo into a directory so the `lmwrap` package can load with `local_files_only=True`.

```bash
uv run lmwrap-download-hf
```

Equivalent:

```bash
uv run python -m lmwrap.utils.download_hf_model
```

Defaults:

- **Preset**: `llama2_7b_chat` unless you pass `--preset` or `--repo-id`.
- **Repo**: from the preset (default `meta-llama/Llama-2-7b-chat-hf`). For preset `gemma4_31b_it` only, omitting `--repo-id` still allows `GEMMA4_REPO_ID`.
- **Destination**: `models/meta-llama-Llama-2-7b-chat-hf` for that preset. For preset `gemma4_31b_it` only, omitting `--local-dir` still allows `GEMMA4_LOCAL_DIR`.

Useful overrides:

```bash
uv run lmwrap-download-hf --local-dir "/path/to/store" --revision main
uv run lmwrap-download-hf --repo-id google/gemma-4-31B-it
uv run lmwrap-download-hf --list-presets
```

Set `HF_TOKEN` if the Hub client needs an explicit token file.

## Inference (CLI)

```bash
uv run lmwrap-infer "Your prompt here." --system "You are a helpful assistant."
```

Equivalent: `uv run python -m lmwrap.app.inference` with the same arguments.

Resolution order for the checkpoint directory:

1. `--model` argument (explicit path).
2. Else `GEMMA4_MODEL`.
3. Else `GEMMA4_LOCAL_DIR` or the relative default `**models/meta-llama-Llama-2-7b-chat-hf**`.

The directory **must contain `config.json`**. Inference does not fall back to pulling the model id from the Hub.

### Load progress

`lmwrap.backend.gemma_backend.load_processor_and_model()` calls `**transformers.utils.logging.enable_progress_bar()**` before loading. Hugging Face Transformers emits **tqdm** bars (for example **Loading weights** over tensor groups during `from_pretrained`). Output goes to **stderr**.

If Hub progress bars were disabled globally, `**enable_progress_bar()`** turns them back on for the duration of that load only.

### Quantization and devices


| Mode            | Effect                                                                                                                                                                                            |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `none`          | Full weights in the default dtype. On macOS with MPS available, the model is loaded with `**device_map={"": "mps"}**` so compute runs on Apple GPU. On CUDA systems, `device_map="auto"` is used. |
| `4bit` / `8bit` | **bitsandbytes** quantized load. **Requires CUDA** in this project. If CUDA is missing, loading raises a clear error (use `none` on Apple Silicon).                                               |


Default when `**GEMMA4_QUANTIZATION` is unset**:

- `**none`** if PyTorch reports MPS as built and available (typical Apple Silicon).
- `**4bit**` otherwise (typical CUDA-oriented workflows).

`PYTORCH_ENABLE_MPS_FALLBACK` is set to `1` before load so operations that lack a Metal kernel can fall back when needed.

### Thinking mode

Gemma 4 supports an instruction-tuned **thinking** path. Control it with flags or environment:

- CLI: `--thinking` / `--no-thinking`.
- If omitted: `GEMMA4_THINKING` set to `1`, `true`, or `yes` enables thinking.

### Other CLI options

- `--max-new-tokens` (default `512`).
- `--system` or `GEMMA4_SYSTEM_PROMPT` for an optional system message.

## Interactive CLI chat (`lmwrap-chat-cli`)

```bash
uv run lmwrap-chat-cli --model ".\models\meta-llama-Llama-2-7b-chat-hf" --qbit 8
```

Equivalent: `uv run python -m lmwrap.app.chat_cli`.

Supports slash commands (`/help`, `/quit`, `/save`, tuning flags, optional debug stats). See `lmwrap-chat-cli --help`.
## Environment variables (summary)


| Variable               | Role                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| `HF_TOKEN`             | Hub token when downloading or if needed elsewhere.                                                   |
| `LMWRAP_ENV_FILE`      | If set, load this `.env` path instead of searching `cwd` or project root.                            |
| `LMWRAP_SKIP_DOTENV`   | Set to `1` / `true` / `yes` to skip `.env` loading.                                                  |
| `OLLAMA_HOST`          | Ollama base URL (default `http://127.0.0.1:11434`).                                                  |
| `GEMMA4_REPO_ID`       | Hub id for download (default `google/gemma-4-31B-it`).                                               |
| `GEMMA4_LOCAL_DIR`     | Folder for snapshots and inference path resolution (default `models/meta-llama-Llama-2-7b-chat-hf`). |
| `GEMMA4_MODEL`         | Explicit directory for inference (overrides base local dir when `--model` is not passed).            |
| `GEMMA4_REVISION`      | Optional revision for downloads.                                                                     |
| `GEMMA4_QUANTIZATION`  | `none`, `4bit`, or `8bit` when you want a fixed default.                                             |
| `GEMMA4_SYSTEM_PROMPT` | Optional system prompt for CLIs (`lmwrap-infer`, `lmwrap-chat-cli`).                         |
| `GEMMA4_THINKING`      | `1` / `true` / `yes` to prefer thinking mode when CLI flags omit it.                                 |


## Notes on `device_map="auto"`

When quantization is `**none**` and **MPS is not** in use (for example Linux with CUDA), the backend uses `**device_map="auto"`** through Hugging Face and Accelerate. That places layers on CUDA when available or CPU otherwise. `**device_map="auto"` alone does not move bitsandbytes-quantized models to MPS.** Quantized checkpoints in this repo are tied to CUDA for that reason.

## GitHub

This folder is an ordinary Git repository (`main`). To publish:

1. Create an empty GitHub repository.
2. `git remote add origin <your-remote-url>`
3. `git push -u origin main`

## Licenses

Upstream **Gemma** weights and terms are governed by Google’s licensing on the Hugging Face model card and **[Apache 2.0](https://huggingface.co/google/gemma-4-31B-it)** metadata for `google/gemma-4-31B-it`. This wrapper repository only ships small scripts; obtain and comply with model terms separately.

## Limitations / future work

- **Metal (MPS) load path:** recent Transformers runs a `**caching_allocator_warmup`** step that allocates one very large FP16 buffer (on the order of full model bytes) before weight copies. CUDA and XPU paths clamp that reservation. **MPS follows the same staging line.** macOS commonly responds with `**RuntimeError: Invalid buffer size`**. `**lmwrap.backend.gemma_backend**` installs a one-time shim on `transformers.modeling_utils.caching_allocator_warmup` so that when any layer in the expanded device map resolves to `**mps**`, the warmup becomes a **no-op**. First-time load may be a bit slower. You can still hit **OOM** if unified memory cannot fit the `**none`** checkpoint.
- **Multimodal** inputs (images, video) require different model classes (`AutoModelForMultimodalLM` and related preprocessing). Current scripts are **text-only**; see the model README on the Hub for image and video snippets.
- A **31B dense** model requires substantial GPU or unified memory in `**none`** mode. Plan hardware accordingly.

