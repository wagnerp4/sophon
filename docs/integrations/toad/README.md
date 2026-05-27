# Mithril + Toad (loose coupling)

Toad ([batrachianai/toad](https://github.com/batrachianai/toad)) is a separate Textual-based launcher for agent CLIs. Mithril now has its own Textual home screen with system stats, directory navigation, a basic shell prompt, and optional chat mode. Toad remains useful as a fuller agent hub with ACP sessions and richer shell/terminal behavior.

## Install Toad (once)

Toad needs **Python 3.14** and its own `uv tool` env (separate from mithril's 3.10 venv).

```powershell
uv python install 3.14
uv tool install -U batrachian-toad --python 3.14
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
```

Verify:

```powershell
toad --version
mithril-cli chat --interface toad --preset llama2_7b_chat
```

If `toad` is installed but not found, set either:

```powershell
$env:MITHRIL_TOAD_BIN = "$env:USERPROFILE\.local\bin\toad.exe"
# or
$env:MITHRIL_TOAD_HOME = "C:\Software\Python\NLP\Repos\Agentic\toad"
```

Upstream Toad targets **Linux/macOS** first. Native Windows launch is blocked by default because the Toad shell often exits with `0xC0000005` (access violation). Mithril sets a UTF-8 console when you opt in. On Windows, prefer mithril's native Textual home/chat app unless you specifically need Toad's agent hub.

```powershell
# supported on Windows
mithril-cli chat --tui --interface textual --preset llama2_7b_chat

# optional native Windows attempt (may still crash)
$env:MITHRIL_TOAD_ALLOW_WINDOWS = "1"
mithril-cli chat --interface toad --preset llama2_7b_chat
```

Prefer WSL/Linux/macOS for Toad itself. Inside Toad, bridge mithril with `!mithril-chat-tui --no-spawn-window ...`.

## Native mithril shell and chat

```powershell
mithril-cli chat --tui --interface textual --preset llama2_7b_chat
```

This opens mithril's home screen first. The model preloads in the background. Use:

- `!dir`, `$pwd`, or `/shell` on Home for shell commands.
- `chat` or `Ctrl+G` on Home to enter chat.
- `Ctrl+H` in Chat to return Home.
- `Ctrl+M` in Chat to focus the model panel.

## Launch Toad from mithril

```powershell
mithril-cli chat --interface toad
```

This opens Toad in a desktop terminal titled **toad** (not the mithril WT profile). On native Windows this is blocked unless `MITHRIL_TOAD_ALLOW_WINDOWS=1`.

## Run mithril chat inside Toad

In Toad shell mode, prefix shell commands with `!`:

```text
!mithril-chat-tui --no-spawn-window --preset llama2_7b_chat
```

Use `--no-spawn-window` so chat stays in the current Toad pane.

## Interface selector

| `--interface` | Host | Use when |
| --- | --- | --- |
| `textual` | mithril Textual TUI (default with `--tui`) | Native mithril chat panels |
| `repl` | plain stdin/stdout REPL | scripting, CI, minimal deps |
| `toad` | external Toad app | agent hub, concurrent sessions, Toad shell |

```powershell
mithril-cli chat --tui --interface textual --preset llama2_7b_chat
mithril-cli chat --interface repl
mithril-cli chat --interface toad
```

## Future ACP agent entry

Toad agents today use the [Agent Client Protocol](https://agentclientprotocol.com/overview/introduction). A first-class Toad tile for mithril would need a small ACP adapter that wraps `mithril-chat-tui`. Until then, the shell bridge above is the supported integration path.

## Terminal profile icon

Install a host profile once:

```powershell
mithril-cli tui-profiles install
```

Windows Terminal gets a **mithril chat** profile with `data/assets/terminal-icon.svg`. macOS and Linux receive helper files under `data/chat_logs/`.
