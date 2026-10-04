# OS surface and deploy

## Role

"Any OS" means one **command and mode language**. Deploy and GPU layout may differ. This file is a constraint on the vision, not a duplicate of [README.md](../../README.md) setup.

## Capability floor

Every host must support:

- UTF-8 I/O (`PYTHONUTF8=1` on Windows child processes)
- `sophon-cli chat --interface repl` without a desktop emulator
- Dashboard / Editor / Chat **if** a TTY of reasonable size exists
- Image preview via half-blocks. Sixel/Kitty optional. Windows skips CSI autodetect (`SOPHON_SKIP_TERMINAL_GRAPHICS=1` in `start-sophon-chat.ps1`)

If a feature needs a graphics protocol, it degrades. If it needs a mouse, it also has a keybind (`.sophon/keybinds.yaml`).

## This machine (WSL + Windows)

| Tree | Path | Owner |
| --- | --- | --- |
| Source | WSL: `.../Signal Processing/NLP/Personal/sophon` | git, Cursor |
| Runtime | `C:\Software\Python\NLP\Personal\sophon` (`SOPHON_WINDOWS_ROOT`) | GPU TUI, PE `.venv` |

```bash
./scripts/deploy-windows.sh
```

rsync keeps Windows `.venv` and `models/`. Profile: Store PowerShell + `scripts\start-sophon-chat.ps1 -Foreground`.

Do not `uv sync` from both OS on the same folder.

If Windows Terminal reports `0x8007010b` (startingDirectory missing), the Windows leftover was deleted. Recreate it with `./scripts/deploy-windows.sh --sync-venv`. See [setup.md](../reference/setup.md).

## Omarchy / Arch (single tree)

On a Linux host with no Windows copy, one checkout is the runtime. Leave `SOPHON_WINDOWS_ROOT` unset.

Prerequisites: `uv`, Python 3.10+, optional NVIDIA + LM Studio / Ollama / local HF weights.

```bash
uv sync --extra tui --extra finetune
cp .env.example .env
sophon-cli chat --no-spawn-window
```

REPL without a desktop emulator:

```bash
sophon-cli chat --interface repl
```

Desktop spawn (optional): if `kitty`, `wezterm`, or `gnome-terminal` is on PATH, `sophon-cli chat` may open a new window. Otherwise stay in the current TTY with `--no-spawn-window`.

Manual kitty / wezterm:

```bash
kitty --title sophon --directory "$PWD" -- .venv/bin/sophon-cli chat --no-spawn-window
wezterm start --cwd "$PWD" -- .venv/bin/sophon-cli chat --no-spawn-window
```

`install_linux_terminal_profile` still writes a snippet under chat logs. It is not a system `.desktop` install.

## Platform matrix

| Action | Windows | WSL | Linux | macOS |
| --- | --- | --- | --- | --- |
| Edit source | yes | yes (preferred here) | yes | yes |
| Textual TUI | WT profile `sophon` | `--windows` or `--linux` | `--no-spawn-window` | Terminal profile stub |
| Desktop spawn | `wt.exe` | `wt.exe -p sophon` | kitty / wezterm / gnome-terminal when present | osascript Terminal |
| Shell tool | PowerShell | bash (Linux TUI) | POSIX | POSIX |
| Autostart | Startup `.lnk` via Store `pwsh` | n/a | later | later |

Textual itself is cross-platform. Sophon is a single CLI (`sophon-cli`) plus extras. The two-tree Windows deploy is optional. Native Linux does not need it.

## Constraints for future OS work

- Paths in skills, `STATE.md`, and RAG must record **which root** (WSL vs Windows) they were built on.
- Editor disk trees (`C:\`, `/`, `/mnt/h`) are host-specific. Do not serialize them into promoted skills.
- Public install is git + `uv` until PyPI. The stub site is [wagnerp4.github.io/sophon](https://wagnerp4.github.io/sophon/). After the first Actions run, set the GitHub repo Pages source to **GitHub Actions**.
- Headless SSH: REPL + CLI eval must work without WT.

## Future direction

1. Linux: `install_linux_terminal_profile` copied to a real `.desktop` or kitty/wezterm snippet. Native spawn already probes kitty / wezterm / gnome-terminal.
2. macOS: replace the chat-log instruction file with a documented iTerm/Ghostty profile.
3. Single-tree mode for machines that only have POSIX (no Windows deploy). `SOPHON_WINDOWS_ROOT` unset. This is the Arch/Omarchy path.
4. Keep `sophon-cli deploy-windows` as the Windows-specific path. Do not invent `deploy-macos` until there is a second runtime tree.
5. PyPI + `uv tool install sophon` as the primary download. Fat `--onedir` binaries later (torch size).
