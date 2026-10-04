# Vision and constraints

## Thesis

sophon is a local-first, terminal-hosted control plane for software work. The long-term shape is not "a chat TUI with extra panels". It is one process that can:

1. **See** host, services, and jobs (dashboard).
2. **Read and change** a tree under review (editor).
3. **Talk to models** with tools (chat).
4. **Bound** what models may do (harness).
5. **Run many agents** without losing the thread (orchestrator).
6. **Keep what worked** as skills and memory only after measurement (self-evolution / Lembas).

The stated product bar is: the terminal is enough to control engineering work on any OS. That bar is a constraint on UI, I/O, and deploy, not a slogan.

## What is already true

- Click CLI (`sophon-cli`) plus a Textual app with Dashboard, Editor, Chat.
- Model backends: Ollama, LM Studio, Hugging Face presets. `SOPHON_CHAT_BACKEND=auto`.
- Retrieval: LEANN index, LightRAG structure, Adaptive-RAG gate. Default corpus: vault + project.
- SQLite chat memory.
- Editor propose/accept (no silent disk writes from LM Studio tools).
- Shell tools: PowerShell on Windows, POSIX shell elsewhere.
- Integrations behind env flags: Obsidian, Zotero, Google, Overleaf, TTS/SST.
- WSL checkout vs Windows deploy tree (`SOPHON_WINDOWS_ROOT`). See [direction/os-and-deploy.md](direction/os-and-deploy.md).

## Hard constraints

### Terminal is the product

- Primary UI is a TTY. Textual is the current host ([TERMINAL.md](TERMINAL.md)). REPL (`--interface repl`) and headless eval must keep working.
- Capability floor: UTF-8, roughly 80x24, no Sixel/Kitty required. Windows Terminal CSI graphics probes are skipped (`SOPHON_SKIP_TERMINAL_GRAPHICS`, default on Windows) because `textual-image` can block forever on console `os.read`.
- Do not make Electron, a web dashboard, or an IDE plugin the only way to complete a task. Those may mirror state later.

### Local-first, explicit network

- Default inference is local (Ollama / LM Studio / HF weights on disk). `/energy local` is that regime.
- Cloud APIs (OpenAI, Anthropic, Google, DeepSeek on the backlog) are `/energy api` with keys in `.env`, never in promoted skills. Spend caps: [direction/energy.md](direction/energy.md).
- Claude Code / Codex CLIs are not an API backend.
- RAG and memory stay on disk unless a storage backend is explicitly configured.

### One venv owner per tree

- `uv sync` from WSL writes POSIX `.venv/bin`. From Windows it writes `.venv/Scripts/*.exe`. They overwrite each other.
- GPU TUI on this machine: Windows PE venv on `C:\Software\Python\NLP\Personal\sophon`. Edit in WSL. Deploy with `./scripts/deploy-windows.sh`.

### Tools are gated and reviewable

- Destructive or ambient tools stay behind `SOPHON_*_TOOLS` (and related) env flags. MCP servers use the same harness table ([integrations/mcp/README.md](integrations/mcp/README.md)).
- Editor tools queue a changeset. Accept / decline / undo / redo is the write path.
- Swarm workers inherit a **subset** of the interactive tool set. Unattended shell and mail send are off until policy says otherwise.

### Self-evolution is not self-rewrite

- Skills and memory may grow. The harness, `pyproject.toml`, and auth files do not auto-edit.
- Promotion requires eval (`/eval model`, `/eval rag`, later skill eval). Failed eval does not merge.
- Finetune on past traces is opt-in and isolated (adapter dir), not an in-place overwrite of the chat preset.

### Headless remains first-class

- CI and scripts use `sophon-benchmark`, `sophon-rag-index`, `sophon-infer`, `sophon-cli chat --interface repl`.
- A TUI-only feature that cannot be driven from CLI is incomplete.

### Secrets and corpus

- `.env` is gitignored. Deploy may copy it to the Windows tree. Do not log tokens.
- Vault / Drive / mail corpora are read tools by default. Writes need an explicit later design (Overleaf write is still TODO).

## Non-goals (until explicitly reversed)

- Replacing the OS window manager or becoming a general multiplexer like tmux.
- Running unbounded swarms on the interactive GPU session (one HF load is already heavy). Subagents share that load or wait. They do not assume a second 12B.
- Kernel-backed Jupyter in the editor (stored outputs only, unless a later spec adds a kernel).
- TradingView or scrape-based price APIs (dashboard already uses Yahoo/Kraken/Open-Meteo).
- Splitting the repo into `src/lembas`, `src/one` before CLI and import migration is specified.

## OS contract

| OS | TUI host | Shell tool | Notes |
| --- | --- | --- | --- |
| Windows | Windows Terminal profile `sophon` (Store `pwsh`) | PowerShell | PE venv. Autostart shortcut optional |
| WSL | Delegates to Windows `sophon-cli.exe` (`--windows`) or `--linux` | bash in Linux TUI | Desktop spawn uses `wt.exe -p sophon` |
| Linux | `--no-spawn-window` (desktop spawn not implemented) | POSIX | Need a native terminal profile later |
| macOS | Profile stub in chat logs dir | POSIX | Terminal.app / iTerm2 / Ghostty |

"Any OS" means the **same command language and modes**, not identical GPUs or identical deploy scripts.

## Loop (intended)

```text
Loop = Agents + Schedule + Context + State Control
Agent = Model + Harness
Skill  = Lembas artifact (prompt, tools allowlist, tests)
State  = one/STATE.md (or successor path) plus SQLite memory
```

Chat modes to implement on top of today's single transcript: **Plan**, **Chat**, **Agent**. Plan must not execute tools. Agent executes only harness-allowed tools. Chat is Q&A with optional retrieve. Agent mode grows a **subagent** tool next: child sessions plus a placement solver (local VRAM vs API inherit). Specification: [direction/subagents.md](direction/subagents.md).

## Naming

| Name | Role |
| --- | --- |
| sophon | Product, package, and harness |
| Lembas | Skills store |
| one | Scheduler / loop / `STATE.md` |
| ResearchAgent | First named agent type on the backlog |

Keep Lembas and one as **docs names** until a package split is scheduled. User-visible strings stay `sophon`.
