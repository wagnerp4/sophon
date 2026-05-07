# lmwrap

Small utilities for working with **[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)** locally using [uv](https://docs.astral.sh/uv/). The tooling covers Hub download, CLI text inference, and a desktop chat UI. All inference paths assume **weights are already on disk** (no runtime Hub fetch for the model).

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

Optional: authenticate for Hub downloads (`HF_TOKEN`, or):

```bash
hf auth login
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/lmwrap/` | Installable package (`lmwrap`): `gemma_backend`, `inference`, `chat_ui`, `main`. |
| `src/lmwrap/gemma_backend.py` | Paths, quantization, `load_processor_and_model`, `generate_response`. |
| `src/lmwrap/inference.py` | CLI entry (`uv run lmwrap-infer` or `python -m lmwrap.inference`). |
| `src/lmwrap/chat_ui.py` | PyQt6 UI (`uv run lmwrap-chat` or `python -m lmwrap.chat_ui`). |
| `scripts/download_gemma4.py` | Download the checkpoint into a local folder (`snapshot_download`). |
| `main.py` | Removed from project root; placeholder remains at `src/lmwrap/main.py`. |

Console scripts from `pyproject.toml`: **`lmwrap-infer`** and **`lmwrap-chat`**.

Downloaded weights are usually kept under `models/` (that directory is listed in `.gitignore` so large files are not committed).

## Downloading weights

The script mirrors the Hub repo into a directory so the `lmwrap` package can load with `local_files_only=True`.

```bash
uv run python scripts/download_gemma4.py
```

Defaults:

- **Repo**: `GEMMA4_REPO_ID` or `google/gemma-4-31B-it`.
- **Destination**: `GEMMA4_LOCAL_DIR` or `models/google-gemma-4-31b-it`.

Useful overrides:

```bash
uv run python scripts/download_gemma4.py --local-dir "/path/to/store" --revision main
uv run python scripts/download_gemma4.py --repo-id google/gemma-4-31B-it
```

Set `HF_TOKEN` if the Hub client needs an explicit token file.

## Inference (CLI)

```bash
uv run lmwrap-infer "Your prompt here." --system "You are a helpful assistant."
```

Equivalent: `uv run python -m lmwrap.inference` with the same arguments.

Resolution order for the checkpoint directory:

1. `--model` argument (explicit path).
2. Else `GEMMA4_MODEL`.
3. Else `GEMMA4_LOCAL_DIR` or the relative default **`models/google-gemma-4-31b-it`**.

The directory **must contain `config.json`**. Inference does not fall back to pulling the model id from the Hub.

### Load progress

`lmwrap.gemma_backend.load_processor_and_model()` calls **`transformers.utils.logging.enable_progress_bar()`** before loading. Hugging Face Transformers emits **tqdm** bars (for example **Loading weights** over tensor groups during `from_pretrained`). Output goes to **stderr**.

If Hub progress bars were disabled globally, **`enable_progress_bar()`** turns them back on for the duration of that load only (the UI path uses its own hooks instead; see below).

### Quantization and devices

| Mode | Effect |
| --- | --- |
| `none` | Full weights in the default dtype. On macOS with MPS available, the model is loaded with **`device_map={"": "mps"}`** so compute runs on Apple GPU. On CUDA systems, `device_map="auto"` is used. |
| `4bit` / `8bit` | **bitsandbytes** quantized load. **Requires CUDA** in this project. If CUDA is missing, loading raises a clear error (use `none` on Apple Silicon). |

Default when **`GEMMA4_QUANTIZATION` is unset**:

- **`none`** if PyTorch reports MPS as built and available (typical Apple Silicon).
- **`4bit`** otherwise (typical CUDA-oriented workflows).

`PYTORCH_ENABLE_MPS_FALLBACK` is set to `1` before load so operations that lack a Metal kernel can fall back when needed.

### Thinking mode

Gemma 4 supports an instruction-tuned **thinking** path. Control it with flags or environment:

- CLI: `--thinking` / `--no-thinking`.
- If omitted: `GEMMA4_THINKING` set to `1`, `true`, or `yes` enables thinking.

### Other CLI options

- `--max-new-tokens` (default `512`).
- `--system` or `GEMMA4_SYSTEM_PROMPT` for an optional system message.

## Desktop chat UI

```bash
uv run lmwrap-chat
```

Equivalent: `uv run python -m lmwrap.chat_ui`.

The window lets you set the **model folder**, **quantization**, optional **system prompt**, then **Load model** (loads on a worker thread). While weights load you get a **`QProgressBar`** (determinate when tqdm reports a total, otherwise busy **indeterminate** mode) plus a label for the tqdm **desc** field (often **Loading weights**). Transformer tqdm output is **suppressed** during UI loads so the terminal is not spammed. After that, use the text field and **Send** for multi-turn chat. **Thinking** and **max new tokens** apply per generation.

The quantization combo initializes from `infer_default_quantization()` so Apple Silicon tends to default to **`none`** for MPS unless you override **`GEMMA4_QUANTIZATION`**.

## Environment variables (summary)

| Variable | Role |
| --- | --- |
| `HF_TOKEN` | Hub token when downloading or if needed elsewhere. |
| `GEMMA4_REPO_ID` | Hub id for download (default `google/gemma-4-31B-it`). |
| `GEMMA4_LOCAL_DIR` | Folder for snapshots and inference path resolution (default `models/google-gemma-4-31b-it`). |
| `GEMMA4_MODEL` | Explicit directory for inference (overrides base local dir when `--model` is not passed). |
| `GEMMA4_REVISION` | Optional revision for downloads. |
| `GEMMA4_QUANTIZATION` | `none`, `4bit`, or `8bit` when you want a fixed default. |
| `GEMMA4_SYSTEM_PROMPT` | Optional system prompt for CLI and UI. |
| `GEMMA4_THINKING` | `1` / `true` / `yes` to prefer thinking mode when CLI flags omit it. |

## Notes on `device_map="auto"`

When quantization is **`none`** and **MPS is not** in use (for example Linux with CUDA), the backend uses **`device_map="auto"`** through Hugging Face and Accelerate. That places layers on CUDA when available or CPU otherwise. **`device_map="auto"` alone does not move bitsandbytes-quantized models to MPS.** Quantized checkpoints in this repo are tied to CUDA for that reason.

## GitHub

This folder is an ordinary Git repository (`main`). To publish:

1. Create an empty GitHub repository.
2. `git remote add origin <your-remote-url>`
3. `git push -u origin main`

## Licenses

Upstream **Gemma** weights and terms are governed by Google’s licensing on the Hugging Face model card and **[Apache 2.0](https://huggingface.co/google/gemma-4-31B-it)** metadata for `google/gemma-4-31B-it`. This wrapper repository only ships small scripts; obtain and comply with model terms separately.

## Limitations / future work

- **Metal (MPS) load path:** recent Transformers runs a **`caching_allocator_warmup`** step that allocates one very large FP16 buffer (on the order of full model bytes) before weight copies. CUDA and XPU paths clamp that reservation. **MPS follows the same staging line.** macOS commonly responds with **`RuntimeError: Invalid buffer size`**. **`lmwrap.gemma_backend`** installs a one-time shim on `transformers.modeling_utils.caching_allocator_warmup` so that when any layer in the expanded device map resolves to **`mps`**, the warmup becomes a **no-op**. First-time load may be a bit slower. You can still hit **OOM** if unified memory cannot fit the **`none`** checkpoint.
- **Multimodal** inputs (images, video) require different model classes (`AutoModelForMultimodalLM` and related preprocessing). Current scripts are **text-only**; see the model README on the Hub for image and video snippets.
- A **31B dense** model requires substantial GPU or unified memory in **`none`** mode. Plan hardware accordingly.
