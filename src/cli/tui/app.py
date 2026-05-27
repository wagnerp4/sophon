from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    MarkdownViewer,
    RichLog,
    Static,
    TextArea,
)

from backend.hf.registry import HF_MODEL_PRESETS
from cli.chat import (
    ChatCliParams,
    ChatIo,
    ModelPickerEntry,
    build_model_picker_entries,
    chat_session_has_weights,
    chat_startup_lines,
    close_chat_session,
    configure_chat_io,
    dispatch_chat_line,
    format_chat_help,
    prepare_chat_session_or_shell,
    reset_chat_io,
    run_chat_generation,
    switch_session_model,
)
from cli.tui_io import TuiStderrSink
from cli.tui_spawn import TuiSessionLog
from utils.device.env_bootstrap import mithril_project_root
from utils.download.hf import download_preset_snapshot, hub_tqdm_bridge_factory


_MODELS_CMD = re.compile(r"^/models(?:\s+(all|local|missing))?\s*$", re.IGNORECASE)
_MODEL_CMD = re.compile(r"^/model\s*$", re.IGNORECASE)
_MARKUP_RE = re.compile(r"\[/?[^\]]+\]")
HOME_TITLE = "mithril"
CHAT_TITLE = "mithril chat"
_MARKDOWN_SUFFIXES = {".md", ".markdown"}


def _set_console_title(title: str) -> None:
    if sys.platform == "win32" and os.environ.get("MITHRIL_TUI_CHILD"):
        try:
            os.system(f"title {title}")
        except Exception:
            pass


def _strip_markup(text: str) -> str:
    return _MARKUP_RE.sub("", text)


def _fmt_bytes(value: object) -> str:
    if not isinstance(value, int) or value < 0:
        return "unknown"
    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= div:
            return f"{value / div:.2f} {label}"
    return f"{value} B"


class ProjectFileActivated(Message):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__()


class ProjectTree(DirectoryTree):
    async def _on_click(self, event: events.Click) -> None:
        await super()._on_click(event)
        if event.chain != 2:
            return
        node = self.cursor_node
        if node is None or node.data is None:
            return
        path = node.data.path
        if not path.is_file():
            return
        self.post_message(ProjectFileActivated(path.resolve()))
        event.stop()


class ModelItemDoubleClicked(Message):
    def __init__(self, entry: ModelPickerEntry) -> None:
        self.entry = entry
        super().__init__()


class SessionReady(Message):
    pass


class SessionFailed(Message):
    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__()


class TurnComplete(Message):
    pass


class ModelListItem(ListItem):
    def __init__(self, entry: ModelPickerEntry) -> None:
        self.entry = entry
        if entry.kind == "header":
            super().__init__(Label(f"[bold]{entry.title}[/bold]"), disabled=True)
            return
        marker = "▸ " if entry.current else "  "
        suffix = "" if entry.local else " [dim]remote[/dim]"
        super().__init__(Label(f"{marker}{entry.title}{suffix}"))

    def on_click(self, event: events.Click) -> None:
        if event.chain >= 2 and self.entry.kind != "header":
            self.post_message(ModelItemDoubleClicked(self.entry))
            event.stop()


class HomeScreen(Screen):
    BINDINGS = [
        Binding("ctrl+g", "open_chat", "Chat", show=True),
        Binding("ctrl+c", "absorb_ctrl_c", show=False, priority=True),
        Binding("escape", "focus_prompt", "Input", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.shell_mode = False
        self._log_lines: list[str] = []
        self._view_mode = "log"
        self._markdown_viewer: MarkdownViewer | None = None

    def compose(self) -> ComposeResult:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        yield Header(show_clock=True)
        with Horizontal(id="home-row"):
            with Vertical(id="home-left"):
                yield Static("Project", id="home-project-title")
                yield ProjectTree(str(app.cwd), id="home-tree")
            with Vertical(id="home-main"):
                yield Static("Loading system stats...", id="home-stats")
                with Vertical(id="home-content"):
                    with Horizontal(id="home-content-toolbar"):
                        yield Static("Shell log", id="home-content-title")
                        yield Button("Copy", id="home-copy", variant="default")
                        yield Button("Back", id="home-back", variant="default")
                    with Vertical(id="home-content-body"):
                        yield TextArea("", id="home-log", read_only=True, show_line_numbers=False)
                yield Input(
                    placeholder="Type !command, $command, /shell, or chat",
                    id="home-prompt",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_stats()
        self.set_interval(30.0, self.refresh_stats)
        self.append_log("Home: Ctrl+G or type chat opens chat. Double-click .md files to preview.")
        self.append_log("Prefix shell commands with ! or $.")
        self.query_one("#home-back", Button).display = False
        self.query_one("#home-prompt", Input).focus()

    def refresh_stats(self) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        try:
            from utils.device.system_check import collect_system_snapshot

            snapshot = collect_system_snapshot(device="all")
        except Exception as exc:
            self.query_one("#home-stats", Static).update(f"System stats unavailable: {exc}")
            return
        host = snapshot.get("host_memory", {})
        devices = snapshot.get("devices", [])
        device_line = "device: cpu"
        if devices:
            first = devices[0]
            if first.get("kind") == "cuda":
                device_line = (
                    f"cuda:{first.get('index')} {first.get('name')} "
                    f"free~={_fmt_bytes(first.get('free_bytes'))}"
                )
            else:
                device_line = f"{first.get('kind')}: {first.get('name', first.get('detail', 'available'))}"
        model_line = app.model_status_text
        cwd_line = f"cwd: {app.cwd}"
        text = "\n".join(
            [
                "mithril home",
                f"python={snapshot.get('python')} torch={snapshot.get('torch')}",
                f"ram={_fmt_bytes(host.get('host_avail_bytes'))} free / {_fmt_bytes(host.get('host_total_bytes'))}",
                device_line,
                f"model: {model_line}",
                cwd_line,
            ]
        )
        self.query_one("#home-stats", Static).update(text)

    def append_log(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            plain = _strip_markup(line)
            if self._log_lines and self._log_lines[-1] == plain:
                continue
            self._log_lines.append(plain)
        self._sync_log_text()

    def _sync_log_text(self) -> None:
        log = self.query_one("#home-log", TextArea)
        log.text = "\n".join(self._log_lines)
        log.scroll_end(animate=False)

    def action_copy_log(self) -> None:
        log = self.query_one("#home-log", TextArea)
        if not log.text.strip():
            self.notify("Nothing to copy.")
            return
        self.app.copy_to_clipboard(log.text)
        self.notify("Copied to clipboard.")

    def action_absorb_ctrl_c(self) -> None:
        self.notify("Use Ctrl+G or type chat to open chat.", timeout=2)

    async def show_markdown(self, path: Path) -> None:
        resolved = path.resolve()
        body = self.query_one("#home-content-body")
        log = self.query_one("#home-log", TextArea)
        log.display = False
        self.query_one("#home-copy", Button).display = False
        self.query_one("#home-back", Button).display = True
        self.query_one("#home-content-title", Static).update(resolved.name)
        if self._markdown_viewer is not None:
            await self._markdown_viewer.remove()
            self._markdown_viewer = None
        viewer = MarkdownViewer(show_table_of_contents=True, id="home-markdown")
        await body.mount(viewer)
        self._markdown_viewer = viewer
        self._view_mode = "markdown"
        try:
            await viewer.go(resolved)
        except Exception as exc:
            await self.show_log()
            self.append_log(f"Could not open {resolved}: {exc}")

    async def show_log(self) -> None:
        if self._markdown_viewer is not None:
            await self._markdown_viewer.remove()
            self._markdown_viewer = None
        self.query_one("#home-log", TextArea).display = True
        self.query_one("#home-copy", Button).display = True
        self.query_one("#home-back", Button).display = False
        self.query_one("#home-content-title", Static).update("Shell log")
        self._view_mode = "log"
        self._sync_log_text()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "home-copy":
            self.action_copy_log()
        elif event.button.id == "home-back":
            self.run_worker(self.show_log, exclusive=True)

    def on_project_file_activated(self, event: ProjectFileActivated) -> None:
        path = event.path
        if path.suffix.lower() not in _MARKDOWN_SUFFIXES:
            return
        self.run_worker(self.show_markdown(path), exclusive=True)

    def action_focus_prompt(self) -> None:
        self.query_one("#home-prompt", Input).focus()

    async def action_open_chat(self) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        await app.open_chat()

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        app.cwd = Path(event.path).resolve()
        self.append_log(f"[dim]cwd -> {app.cwd}[/dim]")
        self.refresh_stats()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        raw = event.value.strip()
        event.input.value = ""
        if not raw:
            return
        if raw.lower() in ("chat", "/chat", "c"):
            self.run_worker(app.open_chat, exclusive=True)
            return
        if raw.lower() in ("/shell", "shell"):
            self.shell_mode = not self.shell_mode
            mode = "on" if self.shell_mode else "off"
            self.append_log(f"[dim]shell mode {mode}[/dim]")
            return
        shell_cmd = raw
        if raw.startswith(("!", "$")):
            shell_cmd = raw[1:].strip()
        elif not self.shell_mode:
            self.append_log("[dim]Prefix commands with ! or $, or type chat to open chat.[/dim]")
            return
        if not shell_cmd:
            return
        self.append_log(f"[bold cyan]$[/bold cyan] {shell_cmd}")
        self.run_worker(lambda: app.run_shell_command(shell_cmd), thread=True, exclusive=True)


class ChatScreen(Screen):
    BINDINGS = [
        Binding("f1", "show_help", "Help", show=True),
        Binding("ctrl+p", "command_palette", "Palette", show=True),
        Binding("ctrl+m", "focus_models", "Models", show=True),
        Binding("ctrl+h", "open_home", "Home", show=True),
        Binding("tab", "cycle_focus", "Focus", show=False),
        Binding("escape", "focus_prompt", "Input", show=True),
        Binding("ctrl+c", "request_quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._focus_target = "prompt"
        self._started = False
        self._busy = False
        self._pending_download: ModelPickerEntry | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-row"):
            with Vertical(id="model-column"):
                yield Static("Models", id="model-panel-title")
                yield ListView(id="model-panel")
                yield Static("↑↓ move · Enter load/download", id="model-panel-hint")
            with Vertical(id="chat-column"):
                yield RichLog(id="transcript", highlight=True, markup=True, wrap=True)
                yield Static("Loading model...", id="status-bar")
                yield Input(
                    placeholder="Message or /command (Enter to send)",
                    id="prompt",
                    disabled=True,
                )
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_from_state()

    def refresh_from_state(self) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        status = self.query_one("#status-bar", Static)
        if app.session_error:
            self._append_system(app.session_error)
            status.update("Failed to load model.")
            return
        if app.session_state is None:
            status.update(f"Loading model... {app.load_status}")
            self.query_one("#prompt", Input).disabled = True
            return
        if not self._started:
            for line in chat_startup_lines(app.session_state):
                self._append_system(line)
            self._append_system(
                "Model panel: Ctrl+M/Tab focus, ↑↓ navigate, Enter load/download. Ctrl+H home."
            )
            self._started = True
        self._refresh_model_panel()
        self.query_one("#prompt", Input).disabled = False
        status.update(self._status_text())
        self.call_after_refresh(self._focus_prompt)

    def update_load_status(self) -> None:
        if self.app.session_state is None:
            self.query_one("#status-bar", Static).update(f"Loading model... {self.app.load_status}")
            return
        self.query_one("#status-bar", Static).update(self._status_text())

    def _status_text(self) -> str:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        st = app.session_state
        if self._busy:
            return app.load_status or "Working..."
        if st is not None and not chat_session_has_weights(st):
            return "No weights loaded — use /models, Ctrl+M panel, or /model-download PRESET."
        if self._focus_target == "models":
            return "Models: ↑↓ navigate · Enter load/download · Tab input"
        return "Ready. Ctrl+M = models · Ctrl+H = home · F1 = /commands."

    def _set_busy(self, active: bool, status: str = "") -> None:
        self._busy = active
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        if status:
            app.load_status = status
        self.query_one("#status-bar", Static).update(self._status_text())
        self.query_one("#prompt", Input).disabled = active
        self.query_one("#model-panel", ListView).disabled = active
        if not active:
            if self._focus_target == "models":
                self.call_after_refresh(self.action_focus_models)
            else:
                self.call_after_refresh(self._focus_prompt)

    def _focus_prompt(self) -> None:
        if self._busy or self.app.session_state is None:
            return
        prompt = self.query_one("#prompt", Input)
        if prompt.disabled:
            return
        self._focus_target = "prompt"
        prompt.focus()
        self.query_one("#status-bar", Static).update(self._status_text())

    def action_focus_prompt(self) -> None:
        self._focus_prompt()

    def action_focus_models(self) -> None:
        if self._busy or self.app.session_state is None:
            return
        panel = self.query_one("#model-panel", ListView)
        if panel.disabled:
            return
        self._focus_target = "models"
        panel.focus()
        self.query_one("#status-bar", Static).update(self._status_text())

    def action_cycle_focus(self) -> None:
        if self._focus_target == "prompt":
            self.action_focus_models()
            return
        self._focus_prompt()

    async def action_open_home(self) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        await app.open_home()

    def action_request_quit(self) -> None:
        self.app.exit()

    def action_show_help(self) -> None:
        self._append_system("Key bindings")
        for key, description in (
            ("f1", "Show slash commands and key bindings"),
            ("ctrl+m", "Focus model panel"),
            ("ctrl+h", "Return home"),
            ("tab", "Switch focus between model panel and input"),
            ("escape", "Return focus to the message input"),
        ):
            self._append_system(f"  {key:<10} {description}")
        self._append_system("Slash commands")
        for line in format_chat_help().splitlines():
            self._append_system(line)
        self._focus_prompt()

    def _append_system(self, text: str) -> None:
        self.query_one("#transcript", RichLog).write(f"[dim]{text}[/dim]")

    def _append_user(self, text: str) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        if app.session_log is not None:
            app.session_log.write("user", text)
        self.query_one("#transcript", RichLog).write(f"[bold cyan]you>[/bold cyan] {text}")

    def _append_assistant(self, text: str) -> None:
        self.query_one("#transcript", RichLog).write("[bold green]assistant>[/bold green]")
        self.query_one("#transcript", RichLog).write(text)

    def append_system_from_app(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            self._append_system(line)

    def append_assistant_from_app(self, text: str) -> None:
        self._append_assistant(text)

    def _refresh_model_panel(self) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        state = app.session_state
        if state is None:
            return
        panel = self.query_one("#model-panel", ListView)
        entries = build_model_picker_entries(state)
        panel.clear()
        if not entries:
            panel.append(ListItem(Label("[dim]no presets[/dim]"), disabled=True))
            return
        highlight_index = 0
        for entry in entries:
            panel.append(ModelListItem(entry))
            if entry.current:
                highlight_index = len(panel.children) - 1
        panel.index = highlight_index

    def _handle_models_panel_command(self, line: str) -> bool:
        if _MODEL_CMD.match(line.strip()) or _MODELS_CMD.match(line.strip()):
            self._append_user(line)
            self._refresh_model_panel()
            self.action_focus_models()
            self._append_system("(model panel focused)")
            return True
        return False

    def _prompt_download(self, entry: ModelPickerEntry) -> None:
        self._pending_download = entry
        repo_id = HF_MODEL_PRESETS[entry.target].repo_id if entry.target in HF_MODEL_PRESETS else entry.detail
        self._append_system(f"Download {entry.target} ({repo_id})? Type y or n.")
        self._focus_prompt()

    def _handle_pending_download_answer(self, line: str) -> bool:
        pending = self._pending_download
        if pending is None:
            return False
        answer = line.strip().lower()
        if answer not in ("y", "yes", "n", "no"):
            self._append_system("Please answer y or n.")
            return True
        self._append_user(line)
        if answer in ("n", "no"):
            self._append_system("(download cancelled)")
            self._pending_download = None
            return True
        self._pending_download = None
        self.run_worker(lambda: self._download_model_worker(pending.target), thread=True, exclusive=True)
        return True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        if app.session_state is None or self._busy:
            return
        line = event.value
        event.input.value = ""
        if line.strip() == "":
            self.call_after_refresh(self._focus_prompt)
            return
        if self._handle_pending_download_answer(line):
            return
        if self._handle_models_panel_command(line):
            return
        self._append_user(line)
        self.run_worker(lambda: self._turn_worker(line), thread=True, exclusive=True)

    def _download_model_worker(self, preset_key: str) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        self.call_from_thread(self._set_busy, True, f"Downloading {preset_key}...")
        try:
            tqdm_class = hub_tqdm_bridge_factory(
                lambda n, total, label: app.progress_from_thread(
                    n,
                    total,
                    f"Downloading {preset_key}: {label}",
                )
            )
            path = download_preset_snapshot(preset_key, tqdm_class=tqdm_class, verbose=True)
            self.call_from_thread(self._append_system, f"(downloaded {preset_key} to {path})")
        except Exception as exc:
            self.call_from_thread(self._append_system, f"(download failed: {exc})")
        finally:
            self.call_from_thread(self._set_busy, False)
            self.call_from_thread(self._refresh_model_panel)

    def _switch_model_worker(self, target: str) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        state = app.session_state
        if state is None:
            return
        self.call_from_thread(self._set_busy, True, f"Loading {target}...")
        try:
            switch_session_model(state, target, on_load_progress=app.progress_from_thread)
        finally:
            self.call_from_thread(self._set_busy, False)
            self.call_from_thread(self._refresh_model_panel)
            self.call_from_thread(self.post_message, TurnComplete())

    def _turn_worker(self, line: str) -> None:
        app = self.app
        assert isinstance(app, MithrilTuiApp)
        state = app.session_state
        if state is None:
            return
        self.call_from_thread(self._set_busy, True, "Generating...")
        try:
            handled, should_generate = dispatch_chat_line(state, line)
            if state.exit_requested:
                self.call_from_thread(self.app.exit)
                return
            if handled and not should_generate:
                if line.strip().lower().startswith("/model"):
                    self.call_from_thread(self._refresh_model_panel)
                return
            if not handled:
                state.messages.append({"role": "user", "content": line.rstrip()})
            run_chat_generation(state)
        finally:
            self.call_from_thread(self._set_busy, False)
            self.call_from_thread(self.post_message, TurnComplete())

    def _activate_model_entry(self, entry: ModelPickerEntry) -> None:
        if self._busy or self.app.session_state is None or entry.kind == "header":
            return
        if not entry.local:
            self._prompt_download(entry)
            return
        if entry.current:
            self._append_system(f"(already using {entry.title})")
            return
        self._append_user(f"/model {entry.target}")
        self.run_worker(lambda: self._switch_model_worker(entry.target), thread=True, exclusive=True)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ModelListItem):
            self._activate_model_entry(item.entry)

    def on_model_item_double_clicked(self, event: ModelItemDoubleClicked) -> None:
        self._activate_model_entry(event.entry)

    def on_turn_complete(self, _event: TurnComplete) -> None:
        if self._focus_target == "models":
            self.action_focus_models()
        else:
            self._focus_prompt()


class MithrilTuiApp(App):
    ENABLE_COMMAND_PALETTE = True
    TITLE = HOME_TITLE
    MODES = {"home": HomeScreen, "chat": ChatScreen}
    CSS = """
Screen {
    layout: vertical;
}

#home-row, #main-row {
    height: 1fr;
}

#home-left {
    width: 36;
    min-width: 28;
    max-width: 48;
    border: solid $secondary;
    background: $surface;
}

#home-project-title, #model-panel-title {
    height: 1;
    padding: 0 1;
    background: $primary;
    color: $text;
    text-style: bold;
}

#home-tree {
    height: 1fr;
}

#home-main, #chat-column {
    width: 1fr;
}

#home-stats {
    height: 7;
    border: solid $primary;
    padding: 0 1;
}

#home-content {
    height: 1fr;
}

#home-content-toolbar {
    height: 1;
    layout: horizontal;
    background: $surface;
    border: solid $primary;
    border-bottom: none;
}

#home-content-title {
    width: 1fr;
    padding: 0 1;
    color: $text-muted;
}

#home-copy, #home-back {
    min-width: 8;
    height: 1;
    border: none;
}

#home-content-body, #transcript {
    height: 1fr;
    border: solid $primary;
    scrollbar-gutter: stable;
}

#home-log, #home-markdown {
    height: 1fr;
    border: none;
}

#home-prompt, #prompt {
    dock: bottom;
    border: tall $accent;
}

#home-prompt:focus-within, #prompt:focus-within {
    border: tall $warning;
}

#model-column {
    width: 36;
    min-width: 28;
    max-width: 48;
    border: solid $secondary;
    background: $surface;
}

#model-panel {
    height: 1fr;
    border-top: solid $secondary;
    border-bottom: solid $secondary;
}

#model-panel:focus-within {
    border-top: solid $warning;
    border-bottom: solid $warning;
}

#model-panel-hint, #status-bar {
    height: 1;
    padding: 0 1;
    color: $text-muted;
}

#status-bar {
    background: $surface;
}
"""

    def __init__(self, params: ChatCliParams, session_log: TuiSessionLog | None) -> None:
        super().__init__()
        self.params = params
        self.session_log = session_log
        self.session_state = None
        self.session_error = ""
        self.load_status = "queued"
        self.model_status_text = "loading..."
        self.cwd = mithril_project_root()
        self._stderr_prev = sys.__stderr__
        self._pending_system_lines: list[str] = []
        self._pending_assistant_lines: list[str] = []

    async def on_mount(self) -> None:
        sys.stderr = TuiStderrSink(self.session_log)  # type: ignore[assignment]
        configure_chat_io(self._wrap_chat_io())
        initial_mode = "chat" if self.params.chat_first else "home"
        self._apply_window_title(CHAT_TITLE if initial_mode == "chat" else HOME_TITLE)
        await self.switch_mode(initial_mode)
        self.run_worker(self._preload_worker, thread=True, exclusive=True)

    def _apply_window_title(self, title: str) -> None:
        self.title = title
        _set_console_title(title)

    def on_unmount(self) -> None:
        if self.session_state is not None:
            close_chat_session(self.session_state)
        reset_chat_io()
        sys.stderr = self._stderr_prev
        if self.session_log is not None:
            self.session_log.close()

    def _wrap_chat_io(self) -> ChatIo:
        def emit(text: str) -> None:
            if self.session_log is not None:
                self.session_log.write("sys", text)
            self.call_from_thread(self.append_system, text)

        def on_assistant(text: str) -> None:
            if self.session_log is not None:
                self.session_log.write("assistant", text)
            self.call_from_thread(self.append_assistant, text)

        return ChatIo(emit=emit, on_assistant=on_assistant)

    def _preload_worker(self) -> None:
        try:
            state = prepare_chat_session_or_shell(self.params, on_load_progress=self.progress_from_thread)
        except Exception as exc:
            self.call_from_thread(self._set_session_failed, f"load failed: {exc}")
            return
        self.call_from_thread(self._set_session_ready, state)

    def progress_from_thread(self, n: int, total: int, label: str) -> None:
        if total > 0:
            status = f"{label} ({n}/{total})"
        else:
            status = label
        self.call_from_thread(self.set_load_status, status)

    def set_load_status(self, status: str) -> None:
        self.load_status = status
        self.model_status_text = status
        screen = self.screen
        if isinstance(screen, HomeScreen):
            screen.refresh_stats()
        elif isinstance(screen, ChatScreen):
            screen.update_load_status()

    def _set_session_ready(self, state) -> None:
        self.session_state = state
        self.session_error = ""
        if chat_session_has_weights(state):
            label = state.preset_key or state.model_path
            self.load_status = f"ready ({label})"
        else:
            self.load_status = "no weights at resolved path (use /models or panel)"
        self.model_status_text = self.load_status
        self.post_message(SessionReady())
        self.refresh_active_screen()

    def _set_session_failed(self, error: str) -> None:
        self.session_error = error
        self.model_status_text = error
        self.post_message(SessionFailed(error))
        self.refresh_active_screen()

    def refresh_active_screen(self) -> None:
        screen = self.screen
        if isinstance(screen, HomeScreen):
            screen.refresh_stats()
        elif isinstance(screen, ChatScreen):
            screen.refresh_from_state()

    async def open_chat(self) -> None:
        self._apply_window_title(CHAT_TITLE)
        await self.switch_mode("chat")
        self.call_after_refresh(self.refresh_active_screen)

    async def open_home(self) -> None:
        self._apply_window_title(HOME_TITLE)
        await self.switch_mode("home")
        self.call_after_refresh(self.refresh_active_screen)

    def append_system(self, text: str) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.append_system_from_app(text)
        else:
            self._pending_system_lines.append(text)

    def append_assistant(self, text: str) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.append_assistant_from_app(text)
        else:
            self._pending_assistant_lines.append(text)

    def run_shell_command(self, command: str) -> None:
        if self._handle_cd(command):
            self.call_from_thread(self._append_home_log, f"[dim]cwd -> {self.cwd}[/dim]")
            return
        if sys.platform == "win32":
            shell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
            if shell:
                argv = [shell, "-NoLogo", "-NoProfile", "-Command", command]
            else:
                argv = ["cmd.exe", "/c", command]
        else:
            shell = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
            code, output = self._run_pty_command([shell, "-lc", command])
            if output.strip():
                self.call_from_thread(self._append_home_log, output.rstrip())
            self.call_from_thread(self._append_home_log, f"[dim](exit {code})[/dim]")
            return
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.cwd),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
        except Exception as exc:
            self.call_from_thread(self._append_home_log, f"[red]{exc}[/red]")
            return
        output = (proc.stdout or "") + (proc.stderr or "")
        if output.strip():
            self.call_from_thread(self._append_home_log, output.rstrip())
        self.call_from_thread(self._append_home_log, f"[dim](exit {proc.returncode})[/dim]")

    def _run_pty_command(self, argv: list[str]) -> tuple[int, str]:
        import pty
        import select

        master_fd, slave_fd = pty.openpty()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.cwd),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
            )
            os.close(slave_fd)
            chunks: list[bytes] = []
            while True:
                ready, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in ready:
                    try:
                        data = os.read(master_fd, 4096)
                    except OSError:
                        data = b""
                    if data:
                        chunks.append(data)
                    else:
                        break
                if proc.poll() is not None and not ready:
                    break
            return proc.wait(), b"".join(chunks).decode("utf-8", errors="replace")
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass

    def _handle_cd(self, command: str) -> bool:
        parts = command.strip().split(maxsplit=1)
        if not parts or parts[0].lower() not in ("cd", "set-location"):
            return False
        if len(parts) == 1:
            self.cwd = Path.home().resolve()
            return True
        target = Path(parts[1].strip('"')).expanduser()
        if not target.is_absolute():
            target = self.cwd / target
        if target.is_dir():
            self.cwd = target.resolve()
            return True
        self.call_from_thread(self._append_home_log, f"[red]not a directory: {target}[/red]")
        return True

    def _append_home_log(self, text: str) -> None:
        screen = self.screen
        if isinstance(screen, HomeScreen):
            screen.append_log(text)


def run_tui_app(params: ChatCliParams, session_log: TuiSessionLog | None) -> None:
    MithrilTuiApp(params, session_log).run()
