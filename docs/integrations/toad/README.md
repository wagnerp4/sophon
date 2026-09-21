# Sophon + Toad (loose coupling)

Toad ([batrachianai/toad](https://github.com/batrachianai/toad)) is a separate Textual-based launcher for agent CLIs. Sophon now has its own Textual home screen with system stats, directory navigation, a basic shell prompt, and optional chat mode. Toad remains useful as a fuller agent hub with ACP sessions and richer shell/terminal behavior.

## Install Toad (once)

Toad needs **Python 3.14** and its own `uv tool` env (separate from sophon's 3.10 venv).

```powershell
uv python install 3.14
uv tool install -U batrachian-toad --python 3.14
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
```

Verify:

```powershell
toad --version
sophon-cli chat --interface toad --preset llama2_7b_chat
```

If `toad` is installed but not found, set either:

```powershell
$env:SOPHON_TOAD_BIN = "$env:USERPROFILE\.local\bin\toad.exe"
# or
$env:SOPHON_TOAD_HOME = "C:\Software\Python\NLP\Repos\Agentic\toad"
```

Upstream Toad targets **Linux/macOS** first. Native Windows launch is blocked by default because the Toad shell often exits with `0xC0000005` (access violation). Sophon sets a UTF-8 console when you opt in. On Windows, prefer sophon's native Textual home/chat app unless you specifically need Toad's agent hub.

```powershell
# supported on Windows
sophon-cli chat --tui --interface textual --preset llama2_7b_chat

# optional native Windows attempt (may still crash)
$env:SOPHON_TOAD_ALLOW_WINDOWS = "1"
sophon-cli chat --interface toad --preset llama2_7b_chat
```

Prefer WSL/Linux/macOS for Toad itself. Inside Toad, bridge sophon with `!sophon-chat-tui --no-spawn-window ...`.

## Native sophon shell and chat

```powershell
sophon-cli chat --tui --interface textual --preset llama2_7b_chat
```

This opens sophon's home screen first. The model preloads in the background. Use:

- `!dir`, `$pwd`, or `/shell` on Home for shell commands.
- `chat` or `Ctrl+G` on Home to enter chat.
- `Ctrl+H` in Chat to return Home.
- `Ctrl+M` or `/models` in Chat to open the model picker.

## Launch Toad from sophon

```powershell
sophon-cli chat --interface toad
```

This opens Toad in a desktop terminal titled **toad** (not the sophon WT profile). On native Windows this is blocked unless `SOPHON_TOAD_ALLOW_WINDOWS=1`.

## Run sophon chat inside Toad

In Toad shell mode, prefix shell commands with `!`:

```text
!sophon-chat-tui --no-spawn-window --preset llama2_7b_chat
```

Use `--no-spawn-window` so chat stays in the current Toad pane.

## Interface selector

| `--interface` | Host | Use when |
| --- | --- | --- |
| `textual` | sophon Textual TUI (default with `--tui`) | Native sophon chat panels |
| `repl` | plain stdin/stdout REPL | scripting, CI, minimal deps |
| `toad` | external Toad app | agent hub, concurrent sessions, Toad shell |

```powershell
sophon-cli chat --tui --interface textual --preset llama2_7b_chat
sophon-cli chat --interface repl
sophon-cli chat --interface toad
```

## Future ACP agent entry

Toad agents today use the [Agent Client Protocol](https://agentclientprotocol.com/overview/introduction). A first-class Toad tile for sophon would need a small ACP adapter that wraps `sophon-chat-tui`. Until then, the shell bridge above is the supported integration path.

## Terminal profile icon

Install a host profile once:

```powershell
sophon-cli tui-profiles install
```

Windows Terminal gets a **sophon chat** profile with `data/assets/terminal-icon.svg`. macOS and Linux receive helper files under `data/chat_logs/`.
