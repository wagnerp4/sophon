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

## Platform matrix

| Action | Windows | WSL | Linux | macOS |
| --- | --- | --- | --- | --- |
| Edit source | yes | yes (preferred here) | yes | yes |
| Textual TUI | WT profile `sophon` | `--windows` or `--linux` | `--no-spawn-window` | Terminal profile stub |
| Desktop spawn | `wt.exe` | `wt.exe -p sophon` | not implemented | osascript Terminal |
| Shell tool | PowerShell | bash (Linux TUI) | POSIX | POSIX |
| Autostart | Startup `.lnk` via Store `pwsh` | n/a | later | later |

Linux desktop spawn raising "not implemented" is a known gap. Closing it means a native terminal profile, not pretending WSL is Linux desktop.

## Constraints for future OS work

- Paths in skills, `STATE.md`, and RAG must record **which root** (WSL vs Windows) they were built on.
- Editor disk trees (`C:\`, `/`, `/mnt/h`) are host-specific. Do not serialize them into promoted skills.
- Installers: `uv` + extras (`tui`, `finetune`, `tts`, `sst`). No requirement for PyInstaller until [TERMINAL.md](../TERMINAL.md) rank for standalone deploy is revisited.
- Headless SSH: REPL + CLI eval must work without WT.

## Future direction

1. Linux: `install_linux_terminal_profile` copied to a real `.desktop` or kitty/wezterm snippet, plus `_spawn_linux` for gnome-terminal/kitty when not WSL.
2. macOS: replace the chat-log instruction file with a documented iTerm/Ghostty profile.
3. Single-tree mode for machines that only have POSIX (no Windows deploy). `SOPHON_WINDOWS_ROOT` unset.
4. Keep `sophon-cli deploy-windows` as the Windows-specific path. Do not invent `deploy-macos` until there is a second runtime tree.
