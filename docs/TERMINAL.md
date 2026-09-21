# Terminal UI & deployment options
Scores: **5** = strong, **3** = adequate, **1** = weak. **Rank** = overall fit for sophon chat + eval (lower is better).
| Rank | Option | Category | Cross-platform | Chat UX | Eval / CI | Custom UI | Testability | Async / streaming | Deps weight | Torch / HF ship | Standalone deploy | Startup | Ecosystem | Scrollback | Multiline | Markdown out | Sophon diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Rich + prompt_toolkit | TUI | 5 | 4 | 5 | 4 | 4 | 4 | 4 | 5 | 2 | 5 | 5 | 4 | 5 | 4 | 5 |
| 2 | Textual | TUI | 5 | 5 | 3 | 5 | 5 | 5 | 3 | 5 | 2 | 4 | 5 | 3 | 4 | 4 | 3 |
| 3 | pip / uv + pipx (entry points) | Deploy | 5 | 1 | 5 | 1 | 5 | 5 | 5 | 5 | 3 | 5 | 5 | 5 | 5 | 1 | 5 |
| 4 | Rich alone | TUI | 5 | 3 | 4 | 3 | 3 | 5 | 5 | 5 | 2 | 5 | 5 | 4 | 2 | 5 | 5 |
| 5 | prompt_toolkit alone | TUI | 5 | 4 | 5 | 3 | 5 | 4 | 5 | 5 | 2 | 5 | 4 | 4 | 5 | 2 | 4 |
| 6 | PyPiTUI | TUI | 4 | 4 | 3 | 4 | 2 | 3 | 4 | 5 | 2 | 4 | 2 | 5 | 4 | 3 | 3 |
| 7 | Hatch / PyPI package | Deploy | 5 | 1 | 5 | 1 | 5 | 5 | 5 | 5 | 3 | 5 | 5 | 5 | 5 | 1 | 5 |
| 8 | PyInstaller (--onedir) | Deploy | 4 | 1 | 4 | 1 | 3 | 5 | 2 | 2 | 5 | 2 | 4 | 5 | 5 | 1 | 4 |
| 9 | PyRatatui | TUI | 4 | 4 | 3 | 5 | 3 | 5 | 2 | 4 | 2 | 4 | 3 | 3 | 3 | 3 | 2 |
| 10 | blessed | TUI | 5 | 3 | 4 | 4 | 2 | 3 | 4 | 5 | 2 | 5 | 3 | 4 | 3 | 2 | 2 |
| 11 | cx_Freeze | Deploy | 4 | 1 | 4 | 1 | 3 | 5 | 2 | 2 | 4 | 3 | 3 | 5 | 5 | 1 | 3 |
| 12 | PyTermGUI | TUI | 4 | 3 | 3 | 4 | 2 | 3 | 4 | 5 | 2 | 4 | 3 | 3 | 3 | 3 | 2 |
| 13 | Nuitka | Deploy | 3 | 1 | 4 | 1 | 3 | 5 | 2 | 2 | 4 | 4 | 3 | 5 | 5 | 1 | 3 |
| 14 | urwid | TUI | 5 | 3 | 3 | 3 | 3 | 3 | 4 | 5 | 2 | 4 | 3 | 3 | 3 | 2 | 2 |
| 15 | py_cui | TUI | 4 | 3 | 3 | 3 | 2 | 2 | 4 | 5 | 2 | 4 | 2 | 3 | 3 | 2 | 2 |
| 16 | Briefcase | Deploy | 4 | 1 | 3 | 1 | 3 | 5 | 1 | 1 | 5 | 2 | 4 | 5 | 5 | 1 | 3 |
| 17 | Plain input / print (current) | TUI | 5 | 2 | 5 | 1 | 4 | 3 | 5 | 5 | 2 | 5 | 5 | 5 | 3 | 1 | 5 |
| 18 | New-terminal launcher only (wt, gnome-terminal, …) | Deploy | 4 | 1 | 5 | 1 | 5 | 5 | 5 | 5 | 1 | 5 | 5 | 5 | 5 | 1 | 5 |
**Legend:** Chat UX = interactive chat quality. Eval / CI = headless benchmark and scripting. Custom UI = panels, themes, layouts. Deps weight = install size and complexity (5 = light). Torch / HF ship = packaging torch/transformers stacks. Standalone deploy = runnable without preinstalled Python/venv. Sophon diff = ease of integrating into current `chat.py` / Click CLI.

## Current Textual Host

`sophon-cli chat --tui --interface textual --preset llama2_7b_chat` opens a desktop terminal and starts a Textual app with three modes (cycle with `Ctrl+H`):

- **Dashboard** (`Ctrl+D`): idle tiles (host, weather, markets, …).
- **Editor** (`Ctrl+E`): project tree for `.py` / `.md` / `.svg` / `.stl` / `.csv` / `.tsv` / `.jsonl` / `.xlsx` / `.ipynb` / `.pdf` / `.png` / `.jpg` / `.webp`. View menu or `Ctrl+Shift+T` cycles Project / Obsidian vault (`SOPHON_VAULT_PATH`) / Zotero / Drive / Bookmarks (when configured) without changing the terminal cwd. View → Disk trees expands a submenu of readable volume roots (`C:\`, `H:\`, …). View → Google accounts connects Gmail/Drive OAuth. File → Search Mail… / Web Search…. Python is full-width. Markdown, SVG, STL, tables, notebooks, PDFs, and images use **source | preview**. STL orbits with arrows or drag. Tables use a DataTable (xlsx is preview-only). Notebooks render stored cells and outputs, with no kernel. PDFs default to extracted page text (PgUp/PgDn). Preview → View → Page raster draws the current page only (Sixel/Kitty when the terminal supports it, otherwise Unicode blocks). Images use the same raster path. Samples live in `data/assets/preview/`. The lower **Terminal** tab runs one PowerShell (Windows) or shell command at a time. `Ctrl+S` saves, `Ctrl+R` refreshes the tree.
- **Chat** (`Ctrl+G`): full-width transcript, status bar, and prompt. `/models` or `Ctrl+M` opens the model picker. With LM Studio, env-enabled tools (speak / Obsidian vault / Zotero / Google mail-Drive-bookmarks-web / local shell) are available together (`/tools`, `/google-status`). Eval and corpus rebuild run here (`/eval`, `/rag-index`).

The model selected by `--preset` preloads in the background while Home is visible. Load and download progress is routed through the status bar/session log so terminal progress bars do not render behind the Textual UI.

Long-term host constraints (any OS, no CSI hang on Windows): [VISION.md](VISION.md), [direction/os-and-deploy.md](direction/os-and-deploy.md).

Use `--chat-first` to keep the old chat-first startup behavior.

If Windows Terminal reports profile settings errors (`0x80070002` / file not found), the profile still points at a missing `.exe`. From the WSL checkout:

```bash
sophon-cli deploy-windows
```

That rewrites the **sophon** profile to Store PowerShell (`%LOCALAPPDATA%\Microsoft\WindowsApps\pwsh.exe`) running `scripts\start-sophon-chat.ps1 -Foreground` on `C:\Software\Python\NLP\Personal\sophon`. Profile only:

```bash
sophon-cli tui-profiles install
```