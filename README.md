<p align="center">
  <img src="data/assets/orodruin.png" alt="Orodruin — the Fiery Mountain" width="100%">
</p>

<h1 align="center">orodruin</h1>

<p align="center">
  Local LLM chat, fine-tuning, and a Textual TUI.<br>
  Named after <b>Orodruin</b>, Sindarin for the Fiery Mountain — Mount Doom.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-8B0000?style=for-the-badge&logo=python&logoColor=white&labelColor=000000" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/model-Gemma%204%2031B-000000?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=8B0000" alt="Gemma 4 31B">
  <img src="https://img.shields.io/badge/tooling-uv-8B0000?style=for-the-badge&labelColor=000000" alt="uv">
  <img src="https://img.shields.io/badge/stack-Transformers-000000?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=8B0000" alt="Transformers">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/-%20-8B0000?style=flat-square" alt="" height="8" width="100%">
</p>

## Requirements

- **Python** 3.10+ ([pyproject.toml](pyproject.toml)).
- **Pillow**, **Torch**, **Torchvision**, **Transformers**, **accelerate**, **bitsandbytes** as declared in [pyproject.toml](pyproject.toml).
- Hugging Face access for gated checkpoints (accept terms on the model card, [HF_TOKEN](https://huggingface.co/docs/hub/security-tokens) or [hf auth login](https://huggingface.co/docs/huggingface_hub/guides/cli)).
- Enough **disk/RAM/GPU memory** for the checkpoint and quantization mode you pick.

<p align="center">
  <img src="https://img.shields.io/badge/-%20-8B0000?style=flat-square" alt="" height="8" width="100%">
</p>

## Setup

One shared repo path (`C:\Software\Python\NLP\orodruin` / `/mnt/c/Software/Python/NLP/orodruin`), but **two different** `.venv` **layouts** depending on who runs `uv sync`:

| Created from             | Layout                          | Use for                                                           |
| ------------------------ | ------------------------------- | ----------------------------------------------------------------- |
| **PowerShell (Windows)** | `.venv/Scripts/orodruin-cli.exe` | Fast GPU loads, `--windows` from WSL, native Windows Terminal TUI |
| **WSL bash**             | `.venv/bin/orodruin-cli`         | Linux-only; slow model I/O on `/mnt/c`                            |

They overwrite each other. Pick one primary environment.

<p align="center">
  <img src="https://img.shields.io/badge/-%20-8B0000?style=flat-square" alt="" height="8" width="100%">
</p>

### Windows (recommended for chat + GPU)

From **PowerShell** (not WSL):

```powershell
cd C:\Software\Python\NLP\orodruin
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
$env:UV_LINK_MODE = "copy"
uv sync --extra tui --extra finetune
```

Separate Textual window from **WSL** (delegates to Windows `.exe`):

```bash
orodruin-cli chat --windows --preset llama2_7b_chat
```

Or from **PowerShell** directly:

```powershell
cd C:\Software\Python\NLP\Personal\orodruin
.\.venv\Scripts\orodruin-cli.exe chat --preset llama2_7b_chat
```

Optional Unsloth backend: add `--extra finetune-unsloth` to `uv sync`.

### WSL-only (slower loads on `/mnt/c`)

```bash
uv sync --extra tui --extra finetune
orodruin-cli chat --linux --preset llama2_7b_chat
```

<p align="center">
  <img src="https://img.shields.io/badge/-%20-8B0000?style=flat-square" alt="" height="8" width="100%">
</p>

## Run

```bash
orodruin-cli chat
orodruin-cli chat --windows
orodruin-cli chat --preset gemma4_12b_it
orodruin-cli chat --interface repl
orodruin-cli chat --interface toad
orodruin-cli finetune --preset llama2_7b_chat --dataset gsm8k_instructions
```

<p align="center">
  <img src="https://img.shields.io/badge/-%20-8B0000?style=flat-square" alt="" height="8" width="100%">
</p>

### Chat backends (Ollama / LM Studio / HF)

Startup resolves `ORODRUIN_CHAT_BACKEND` (default `auto`):

1. **Ollama** if `OLLAMA_HOST` responds **and** at least one model is listed
2. else **LM Studio** if OpenAI-compatible `/v1/models` responds **and** at least one model is listed
3. else **Hugging Face** presets / local weights

Startup picks a default from whatever has models (ollama → lmstudio → hf), but `/models` (TUI: overlay picker, also Ctrl+M) always lists **all** sources. Selecting an entry switches backend automatically (`ollama:…`, `lmstudio:…`, or HF preset). `/backend` is optional.

In chat: `/backend`, `/models`, `/model NAME`. Server backends need no Hub download.

```bash
ORODRUIN_CHAT_BACKEND=auto
OLLAMA_HOST=http://127.0.0.1:11434
ORODRUIN_LM_STUDIO_HOST=http://127.0.0.1:1234/v1
```

<p align="center">
  <img src="https://img.shields.io/badge/-%20-8B0000?style=flat-square" alt="" height="8" width="100%">
</p>

### TTS (Pipecat default)

Default speech backend is **Pipecat Kokoro** (local ONNX, few knobs). Install extras:

```powershell
# close orodruin-cli / TUI first if Windows locks Scripts\*.exe
uv sync --extra tui --extra tts
```

```bash
ORODRUIN_TTS_BACKEND=pipecat
ORODRUIN_TTS_SPEAKER=am_adam
ORODRUIN_TTS_TOOL=1
```

Chat: `/speak TEXT`, `/tts on|off`, `/tts-backend pipecat|hf-qwen-custom-voice|ollama`. With LM Studio, models can call the `speak` tool when `ORODRUIN_TTS_TOOL=1`. Qwen CustomVoice remains available via `/tts-backend hf-qwen-custom-voice`.

Generation knobs (HF and LM Studio / Ollama): `/params`, `/temp`, `/top-p`, `/max`, `/seed`, `/rep`. Server context window is set when the model is loaded in LM Studio / Ollama. `/max` is the per-reply token budget. LM Studio tool-loop rounds default to unlimited (`/tool-rounds N`, `/unlimited`).

<p align="center">
  <img src="https://img.shields.io/badge/-%20-8B0000?style=flat-square" alt="" height="8" width="100%">
</p>

### Obsidian vault (LM Studio tools)

With Obsidian **Local REST API** running, orodruin can expose read-only vault tools to LM Studio chat (`vault_search`, `vault_list`, `vault_read`, `vault_recent`). Prefer adding these to the repo `.env` (loaded on startup), then restart chat:

```text
ORODRUIN_OBSIDIAN_TOOLS=1
ORODRUIN_OBSIDIAN_API_URL=https://127.0.0.1:27124
ORODRUIN_OBSIDIAN_API_KEY=paste-from-plugin
```

One-shot in **PowerShell** (current session only):

```powershell
$env:ORODRUIN_OBSIDIAN_TOOLS = "1"
$env:ORODRUIN_OBSIDIAN_API_URL = "https://127.0.0.1:27124"
$env:ORODRUIN_OBSIDIAN_API_KEY = "paste-from-plugin"
```

Check with `/obsidian-status`. See `docs/integrations/obsidian/README.md` for LM Studio native MCP (`mcp.json`) setup (same vault, app chat UI).

<p align="center">
  <img src="https://img.shields.io/badge/-%20-8B0000?style=flat-square" alt="" height="8" width="100%">
</p>