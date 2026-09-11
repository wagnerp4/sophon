from __future__ import annotations

import datetime as _dt
import os
import re
import sys
import time
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from backend.chat_resolve import is_server_backend
from backend.hf.registry import HF_MODEL_PRESETS, resolve_chat_startup_model, resolve_preset_dir
from cli.chat import (
    ChatCliParams,
    ChatIo,
    ModelPickerEntry,
    build_model_picker_entries,
    build_model_picker_entries_for_root,
    chat_session_ready,
    close_chat_session,
    configure_chat_io,
    dispatch_chat_line,
    format_chat_help,
    prepare_chat_session_or_shell,
    reset_chat_io,
    run_chat_generation,
    switch_session_model,
    transcribe_audio_paths,
)
from cli.chat_display import (
    TranscriptLine,
    chat_banner_transcript_lines,
    format_assistant_body_line,
    format_assistant_turn_header,
    format_system_line,
    format_user_turn_line,
)
from cli.host.session_log import TuiSessionLog
from cli.tui.io import TuiStderrSink
from cli.tui.paste_drop import (
    build_message_with_attachments,
    describe_dropped_paths,
    extract_dropped_paths,
    format_paths_for_prompt,
    paths_still_in_text,
    split_audio_paths,
)
from cli.tui.screens.dashboard import DashboardScreen
from cli.tui.screens.nav import (
    CHAT_TITLE,
    DASHBOARD_TITLE,
    EDITOR_TITLE,
    MODE_CYCLE,
    NAV_BINDINGS,
    WORKSHOP_TITLE,
    ModeNavigationMixin,
    compose_nav_bar,
)
from cli.tui.screens.editor import EditorScreen
from cli.tui.screens.workshop import WorkshopScreen
from utils.device.env_bootstrap import orodruin_project_root
from utils.download.hf import (
    capture_hub_download_logs,
    capture_hub_user_warnings,
    download_preset_snapshot,
    format_byte_progress_pair,
    format_download_progress,
    format_idle_download_hint,
    format_progress_bar,
    hub_tqdm_bridge_factory,
    is_file_count_progress,
    normalize_hub_phase,
    split_hub_progress_label,
    throttled_progress_callback,
    watch_local_download_bytes,
)


_MODELS_CMD = re.compile(r"^/models(?:\s+(all|local|missing))?\s*$", re.IGNORECASE)


def _set_console_title(title: str) -> None:
    if sys.platform == "win32" and os.environ.get("ORODRUIN_TUI_CHILD"):
        try:
            os.system(f"title {title}")
        except Exception:
            pass


def _format_action_elapsed(seconds: float) -> str:
    total = int(max(0, seconds))
    if total < 3600:
        mins, secs = divmod(total, 60)
        if mins == 0:
            return f"{secs}s"
        return f"{mins}m {secs:02d}s"
    hours, rem = divmod(total, 3600)
    mins, _ = divmod(rem, 60)
    return f"{hours}h {mins}m"


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


class FilesDropped(Message):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        super().__init__()


class ChatPromptInput(Input):
    def _on_paste(self, event: events.Paste) -> None:
        paths = extract_dropped_paths(event.text)
        if paths:
            event.stop()
            self.post_message(FilesDropped(paths))
            return
        super()._on_paste(event)


class LoadProgress(Message):
    def __init__(self, status: str, *, n: int = 0, total: int = 0, phase: str = "") -> None:
        self.status = status
        self.n = n
        self.total = total
        self.phase = phase
        super().__init__()


class HubLogLine(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class ModelListItem(ListItem):
    _NON_SELECTABLE = frozenset({"header", "group", "subgroup"})

    def __init__(self, entry: ModelPickerEntry) -> None:
        self.entry = entry
        if entry.kind == "header":
            super().__init__(Label(f"[bold]{entry.title}[/bold]"), disabled=True)
            return
        if entry.kind == "group":
            super().__init__(Label(f"[bold dim]{entry.title}[/bold dim]"), disabled=True)
            return
        if entry.kind == "subgroup":
            super().__init__(Label(f"[dim]  {entry.title}[/dim]"), disabled=True)
            return
        if entry.local:
            marker = "▸ " if entry.current else "  "
        else:
            marker = "    ▸ " if entry.current else "    "
        suffix = "" if entry.local else " [dim]remote[/dim]"
        super().__init__(Label(f"{marker}{entry.title}{suffix}"))

    def on_click(self, event: events.Click) -> None:
        if event.chain >= 2 and self.entry.kind not in self._NON_SELECTABLE:
            self.post_message(ModelItemDoubleClicked(self.entry))
            event.stop()


class ModelPickerScreen(ModalScreen[ModelPickerEntry | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Close", show=True),
        Binding("ctrl+m", "cancel", "Close", show=False),
    ]

    def __init__(self, entries: list[ModelPickerEntry]) -> None:
        super().__init__()
        self._entries = entries

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-dialog"):
            yield Static("Models — ↑↓ move · Enter load/download · Esc close", id="model-picker-title")
            yield ListView(id="model-picker-list")
            yield Static("Click or Enter to select", id="model-picker-hint")

    def on_mount(self) -> None:
        panel = self.query_one("#model-picker-list", ListView)
        panel.clear()
        if not self._entries:
            panel.append(ListItem(Label("[dim]no models[/dim]"), disabled=True))
            return
        highlight_index = 0
        for entry in self._entries:
            panel.append(ModelListItem(entry))
            if entry.current:
                highlight_index = len(panel.children) - 1
        panel.index = highlight_index
        panel.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ModelListItem) and item.entry.kind not in ModelListItem._NON_SELECTABLE:
            self.dismiss(item.entry)

    def on_model_item_double_clicked(self, event: ModelItemDoubleClicked) -> None:
        if event.entry.kind not in ModelListItem._NON_SELECTABLE:
            self.dismiss(event.entry)


class ChatScreen(ModeNavigationMixin, Screen):
    BINDINGS = [
        Binding("f1", "show_help", "Help", show=True),
        Binding("ctrl+p", "command_palette", "Palette", show=True),
        Binding("ctrl+m", "open_models", "Models", show=True),
        *NAV_BINDINGS,
        Binding("ctrl+y", "copy_transcript", "Copy", show=True),
        Binding("ctrl+l", "listen", "Listen", show=True, priority=True),
        Binding("ctrl+shift+t", "focus_transcript", "Log", show=True),
        Binding("tab", "cycle_focus", "Focus", show=False),
        Binding("escape", "focus_prompt", "Input", show=True),
        Binding("ctrl+c", "absorb_ctrl_c", show=False, priority=True),
        Binding("ctrl+q", "request_quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._focus_target = "prompt"
        self._started = False
        self._busy = False
        self._pending_download: ModelPickerEntry | None = None
        self._transcript_plain: list[str] = []
        self._turn_counter = 0
        self._session_started_at = _dt.datetime.now()
        self._picker_open = False
        self._spinner_i = 0
        self._activity_kind = ""
        self._dropped_paths: list[Path] = []

    def _query_widget(self, selector: str, widget_type: type):
        try:
            return self.query_one(selector, widget_type)
        except NoMatches:
            return None

    def _format_status_text(self, text: str) -> str:
        app = self.app
        if isinstance(app, OrodruinTuiApp):
            return app.format_with_elapsed(text)
        return text

    def _update_status_bar(self, text: str) -> None:
        status = self._query_widget("#status-bar", Static)
        if status is not None:
            status.update(self._format_status_text(text))

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield from compose_nav_bar("chat")
        with Vertical(id="chat-column"):
            yield Static("", id="chat-banner")
            with Horizontal(id="chat-toolbar"):
                yield Static("chat", id="chat-toolbar-title")
                yield Button("Copy", id="chat-copy", variant="default")
            yield RichLog(id="transcript", highlight=True, markup=True, wrap=True, auto_scroll=True)
            with Vertical(id="progress-panel"):
                yield Static("", id="download-phase")
                yield Static("", id="download-bar")
            yield Static("Loading model...", id="status-bar")
            yield ChatPromptInput(
                placeholder="Message, /command, or drop a file (Enter to send)",
                id="prompt",
                disabled=True,
            )
        yield Footer()

    def on_mount(self) -> None:
        panel = self._query_widget("#progress-panel", Vertical)
        if panel is not None:
            panel.display = False
        self.refresh_from_state()
        self.set_interval(0.25, self._tick_action_elapsed)

    def _tick_action_elapsed(self) -> None:
        app = self.app
        if not isinstance(app, OrodruinTuiApp):
            return
        if self._busy:
            if app.load_progress_total > 0 or app.load_progress_n > 0:
                self.update_load_status()
            else:
                self._show_activity_panel()
            return
        if app._action_started_at is None:
            return
        if app.session_state is None:
            self.update_load_status()

    def refresh_from_state(self) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        if self._query_widget("#status-bar", Static) is None:
            return
        if app.session_error:
            self._append_system(app.session_error)
            self._update_status_bar("Failed to load model.")
            return
        if app.session_state is None:
            self._update_status_bar(f"Loading model... {app.load_status}")
            prompt = self._query_widget("#prompt", Input)
            if prompt is not None:
                prompt.disabled = True
            return
        self._render_chat_banner()
        if not self._started:
            log_path = str(app.session_log.path) if app.session_log is not None else None
            stamp = self._session_started_at.strftime("%Y-%m-%d %H:%M:%S")
            self._append_system(f"session started {stamp}")
            if log_path:
                self._append_system(f"log {log_path}")
            self._started = True
        pending_sys = app._pending_system_lines
        if pending_sys:
            for text in list(pending_sys):
                self._append_system(text)
            pending_sys.clear()
        pending_asst = app._pending_assistant_lines
        if pending_asst:
            for text in list(pending_asst):
                self._append_assistant(text)
            pending_asst.clear()
        self._hide_progress_panel()
        prompt = self._query_widget("#prompt", Input)
        if prompt is not None:
            prompt.disabled = False
        self._update_status_bar(self._status_text())
        self.call_after_refresh(self._focus_prompt)

    def _banner_width(self) -> int:
        banner = self._query_widget("#chat-banner", Static)
        if banner is not None and banner.size.width > 8:
            return max(int(banner.size.width) - 2, 40)
        column = self._query_widget("#chat-column", Vertical)
        if column is not None and column.size.width > 8:
            return max(int(column.size.width) - 4, 40)
        if self.size.width:
            return max(int(self.size.width) - 6, 40)
        return 80

    def _render_chat_banner(self) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        banner = self._query_widget("#chat-banner", Static)
        state = app.session_state
        if banner is None or state is None:
            return
        lines = chat_banner_transcript_lines(
            state,
            log_path=None,
            started_at=self._session_started_at,
            for_markup=True,
            width=self._banner_width(),
        )
        markup = "\n".join(line.markup for line in lines if line.plain or line.markup)
        banner.update(markup)

    def update_load_status(self) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        if app.session_state is not None and not self._busy:
            self._hide_progress_panel()
            self._update_status_bar(self._status_text())
            return
        if (self._busy or app.session_state is None) and (
            app.load_progress_total > 0 or app.load_progress_n > 0
        ):
            self._update_progress_panel(
                app.load_progress_n,
                app.load_progress_total,
                app.load_progress_phase or app.load_status,
            )
            return
        if self._busy:
            self._show_activity_panel()
            return
        if app.session_state is None:
            text = app.model_status_text or f"Loading model... {app.load_status}"
            if app.load_progress_total > 0:
                text = f"{app.load_progress_phase or app.load_status} — {app.load_progress_n}/{app.load_progress_total}"
                text = app.format_with_elapsed(text)
            self._update_status_bar(text)
            return
        self._update_status_bar(self._status_text())

    def _hide_progress_panel(self) -> None:
        panel = self._query_widget("#progress-panel", Vertical)
        if panel is not None:
            panel.display = False

    def _activity_label(self) -> str:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        frame = frames[self._spinner_i % len(frames)]
        self._spinner_i += 1
        base = (app.load_status or self._activity_kind or "Working").strip() or "Working"
        return app.format_with_elapsed(f"{frame} {base}")

    def _show_activity_panel(self) -> None:
        kind = (self._activity_kind or "working").lower()
        if (
            "generat" in kind
            or "thinking" in kind
            or "speak" in kind
            or "transcrib" in kind
            or "listen" in kind
        ):
            self._hide_progress_panel()
            self._update_status_bar(self._activity_label())
            return
        panel = self._query_widget("#progress-panel", Vertical)
        phase_widget = self._query_widget("#download-phase", Static)
        bar_widget = self._query_widget("#download-bar", Static)
        if panel is None or phase_widget is None or bar_widget is None:
            self._update_status_bar(self._activity_label())
            return
        panel.display = True
        label = self._activity_label()
        phase_widget.update(label)
        if "download" in kind:
            hint = "download in progress"
        elif "load" in kind:
            hint = "loading weights / switching model"
        else:
            hint = "in progress · server calls are not cancelled by Esc"
        bar_widget.update(hint)
        status = self._query_widget("#status-bar", Static)
        if status is not None:
            status.update("")

    def _update_progress_panel(self, n: int, total: int, phase: str) -> None:
        panel = self._query_widget("#progress-panel", Vertical)
        phase_widget = self._query_widget("#download-phase", Static)
        bar_widget = self._query_widget("#download-bar", Static)
        if panel is None or phase_widget is None or bar_widget is None:
            return
        panel.display = True
        app = self.app
        clean_phase = normalize_hub_phase(phase)
        if clean_phase.lower().startswith("downloading "):
            clean_phase = clean_phase.split(":", 1)[-1].strip()
        if isinstance(app, OrodruinTuiApp):
            clean_phase = app.format_with_elapsed(clean_phase)
        phase_widget.update(clean_phase)
        if is_file_count_progress(n, total, phase):
            if total > 0:
                bar = format_progress_bar(n, total)
                bar_widget.update(f"[{bar}] {n} / {total} files")
            else:
                bar_widget.update("enumerating repo files on Hub...")
        elif total > 0:
            bar = format_progress_bar(n, total)
            detail = format_byte_progress_pair(n, total)
            bar_widget.update(f"[{bar}] {detail}")
        elif n > 0:
            bar_widget.update(format_byte_progress_pair(n, total))
        elif "weight" in clean_phase.lower() or "tensor" in clean_phase.lower():
            bar_widget.update(
                "initializing quantized load — shard bar follows; "
                "/mnt/c checkpoints are slow (copy to ~/ if this stalls)"
            )
        else:
            bar_widget.update(format_idle_download_hint())
        status = self._query_widget("#status-bar", Static)
        if status is not None:
            status.update("")

    def _status_text(self) -> str:
        from cli.chat_display import chat_ready_status_line

        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        st = app.session_state
        if self._busy:
            return self._activity_label()
        if st is not None and not chat_session_ready(st):
            if is_server_backend(st.backend_id):
                return f"No {st.backend_id} model — /models or Ctrl+M, then Enter to load."
            return "No weights loaded — /models or Ctrl+M, or /model-download PRESET."
        if self._focus_target == "transcript":
            return "Chat: select text · Copy button or Ctrl+Y · Esc input"
        if st is not None:
            return chat_ready_status_line(st)
        return "Ready. /models or Ctrl+M · Ctrl+D/W/E/G panes · F1 = /commands."

    def _set_busy(self, active: bool, status: str = "") -> None:
        self._busy = active
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        if active:
            app.start_action_timer(reset=True)
            self._spinner_i = 0
            self._activity_kind = status or "Working"
        else:
            app.stop_action_timer()
            self._activity_kind = ""
        if status:
            app.load_status = status
            app.load_progress_phase = status
        if active and (app.load_progress_total > 0 or status):
            if app.load_progress_total > 0 or app.load_progress_n > 0:
                self.update_load_status()
            else:
                self._show_activity_panel()
        else:
            if not active:
                self._hide_progress_panel()
            self._update_status_bar(self._status_text())
        prompt = self._query_widget("#prompt", Input)
        if prompt is not None:
            prompt.disabled = active
        if not active:
            self.call_after_refresh(self._focus_prompt)

    def _focus_prompt(self) -> None:
        if self._busy or self.app.session_state is None:
            return
        prompt = self._query_widget("#prompt", Input)
        if prompt is None or prompt.disabled:
            return
        self._focus_target = "prompt"
        prompt.focus()
        self._update_status_bar(self._status_text())

    def action_focus_prompt(self) -> None:
        self._focus_prompt()

    def action_open_models(self) -> None:
        if self._busy or self.app.session_state is None or self._picker_open:
            return
        self._open_model_picker()

    def action_cycle_focus(self) -> None:
        if self._focus_target == "prompt":
            self.action_focus_transcript()
            return
        self._focus_prompt()

    def action_focus_transcript(self) -> None:
        if self._busy or self.app.session_state is None:
            return
        self._focus_target = "transcript"
        transcript = self._query_widget("#transcript", RichLog)
        if transcript is not None:
            transcript.focus()
        self._update_status_bar(self._status_text())

    def action_listen(self) -> None:
        if self._busy or self.app.session_state is None:
            return
        self.run_worker(lambda: self._turn_worker("/listen"), thread=True, exclusive=True)

    def action_copy_transcript(self) -> None:
        payload = "\n".join(self._transcript_plain)
        if not payload.strip():
            self.notify("Nothing to copy.")
            return
        self.app.copy_to_clipboard(payload)
        self.notify("Copied chat to clipboard.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.handle_nav_button(event.button.id):
            return
        if event.button.id == "chat-copy":
            self.action_copy_transcript()

    def action_absorb_ctrl_c(self) -> None:
        self.notify("Ctrl+Y or the Copy button copies chat text. /quit or Ctrl+Q exits.", timeout=4)

    def action_request_quit(self) -> None:
        self.app.exit()

    def action_show_help(self) -> None:
        self._append_system("Key bindings")
        for key, description in (
            ("f1", "Show slash commands and key bindings"),
            ("ctrl+m", "Open model picker (/models)"),
            ("ctrl+d", "Open dashboard"),
            ("ctrl+w", "Open workshop"),
            ("ctrl+e", "Open editor (project tree · .py / .md / .svg / .stl / tables / .ipynb / .pdf)"),
            ("right", "Editor: accept autocomplete suggestion"),
            ("ctrl+b", "Editor: toggle project pane"),
            ("ctrl+j", "Editor: toggle lower panel"),
            ("ctrl+h", "Cycle dashboard / workshop / editor / chat"),
            ("ctrl+shift+t", "Focus chat log"),
            ("ctrl+l", "Record from the microphone into the prompt (/listen)"),
            ("ctrl+y", "Copy full chat log to clipboard"),
            ("copy", "Toolbar Copy button — same as Ctrl+Y"),
            ("ctrl+q", "Exit the TUI (/quit also works)"),
            ("tab", "Switch focus between chat log and input"),
            ("escape", "Return focus to the message input"),
        ):
            self._append_system(f"  {key:<10} {description}")
        self._append_system("Slash commands")
        for line in format_chat_help().splitlines():
            self._append_system(line)
        self._focus_prompt()

    def _write_transcript(self, line: TranscriptLine) -> None:
        self._transcript_plain.append(line.plain)
        log = self._query_widget("#transcript", RichLog)
        if log is not None:
            if line.plain:
                log.write(line.markup)
            else:
                log.write("")

    def _append_system(self, text: str) -> None:
        for chunk in text.splitlines() or [""]:
            self._write_transcript(format_system_line(chunk, for_markup=True))

    def _append_user(self, text: str) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        if app.session_log is not None:
            app.session_log.write("user", text)
        self._turn_counter += 1
        self._write_transcript(
            format_user_turn_line(text, turn=self._turn_counter, for_markup=True)
        )

    def _append_assistant(self, text: str) -> None:
        self._write_transcript(
            format_assistant_turn_header(turn=self._turn_counter, for_markup=True)
        )
        for chunk in text.splitlines() or [""]:
            self._write_transcript(format_assistant_body_line(chunk, for_markup=True))

    def append_system_from_app(self, text: str) -> None:
        if "conversation cleared" in text.lower():
            self._turn_counter = 0
        for line in text.splitlines() or [""]:
            self._append_system(line)

    def append_assistant_from_app(self, text: str) -> None:
        self._append_assistant(text)

    def _model_picker_entries(self, app: "OrodruinTuiApp") -> list[ModelPickerEntry]:
        state = app.session_state
        if state is not None:
            return build_model_picker_entries(state)
        root = app.cwd
        current_preset = app.params.preset
        current_path: Path | None = None
        if app.params.model:
            current_path = Path(app.params.model).expanduser().resolve()
        elif current_preset is None:
            preset_key, resolved_dir = resolve_chat_startup_model(
                preset=None,
                model=None,
                cwd=root,
            )
            current_preset = preset_key
            if (resolved_dir / "config.json").is_file():
                current_path = resolved_dir.resolve()
        return build_model_picker_entries_for_root(
            root,
            current_preset=current_preset,
            current_path=current_path,
        )

    def _open_model_picker(self) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        if self._picker_open:
            return
        self._picker_open = True

        def on_close(entry: ModelPickerEntry | None) -> None:
            self._picker_open = False
            if entry is not None:
                self._activate_model_entry(entry)
            self.call_after_refresh(self._focus_prompt)

        self.app.push_screen(ModelPickerScreen(self._model_picker_entries(app)), on_close)

    def _handle_models_command(self, line: str) -> bool:
        if not _MODELS_CMD.match(line.strip()):
            return False
        self._append_user(line)
        self._append_system("(model picker — ↑↓ / click · Enter · Esc)")
        self._open_model_picker()
        return True

    def _prompt_download(self, entry: ModelPickerEntry) -> None:
        self._pending_download = entry
        target = entry.target
        if target.lower().startswith("hf:"):
            target = target[3:].strip()
        repo_id = HF_MODEL_PRESETS[target].repo_id if target in HF_MODEL_PRESETS else entry.detail
        self._append_system(f"Download {target} ({repo_id})? Type y or n.")
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
        preset = pending.target
        if preset.lower().startswith("hf:"):
            preset = preset[3:].strip()
        self.run_worker(lambda: self._download_model_worker(preset), thread=True, exclusive=True)
        return True

    def on_files_dropped(self, event: FilesDropped) -> None:
        if self._busy or self.app.session_state is None:
            return
        paths = list(event.paths)
        if not paths:
            return
        audio, other = split_audio_paths(paths)
        if audio:
            self._dropped_paths = other
            names = ", ".join(p.name for p in audio)
            self._append_system(f"(sst dropping {len(audio)} audio file(s): {names})")
            self.run_worker(
                lambda: self._transcribe_drop_worker(audio, other),
                thread=True,
                exclusive=True,
            )
            return
        self._dropped_paths = paths
        prompt = self._query_widget("#prompt", Input)
        if prompt is None:
            return
        inserted = format_paths_for_prompt(paths)
        existing = prompt.value.strip()
        if existing and not paths_still_in_text(existing, paths):
            prompt.value = f"{existing} {inserted}".strip()
        else:
            prompt.value = inserted
        prompt.cursor_position = len(prompt.value)
        self._append_system(f"({describe_dropped_paths(paths)})")
        self._update_status_bar(
            f"Dropped {len(paths)} path(s). Add a note or press Enter to attach."
        )
        self.call_after_refresh(self._focus_prompt)

    def _insert_prompt_text(self, text: str) -> None:
        prompt = self._query_widget("#prompt", Input)
        if prompt is None:
            return
        existing = prompt.value.strip()
        body = text.strip()
        if not body:
            return
        if existing:
            prompt.value = f"{existing} {body}".strip()
        else:
            prompt.value = body
        prompt.cursor_position = len(prompt.value)
        self.call_after_refresh(self._focus_prompt)

    def insert_dictated_text(self, text: str) -> None:
        self._insert_prompt_text(text)
        self._append_system("(dictated into prompt · edit or Enter to send)")
        self.call_after_refresh(self._focus_prompt)

    def _transcribe_drop_worker(self, audio: list[Path], other: list[Path]) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        state = app.session_state
        if state is None:
            return
        app.call_from_thread(self._set_busy, True, "Transcribing…")
        try:
            text = transcribe_audio_paths(state, audio)
        except Exception as exc:
            app.call_from_thread(self._append_system, f"(sst failed: {exc})")
            app.call_from_thread(self._set_busy, False)
            return
        if text.strip():
            app.call_from_thread(self._insert_prompt_text, text.strip())
        if other:
            inserted = format_paths_for_prompt(other)
            app.call_from_thread(self._insert_prompt_text, inserted)
            app.call_from_thread(self._append_system, f"({describe_dropped_paths(other)})")
        app.call_from_thread(self._set_busy, False)
        app.call_from_thread(self._focus_prompt)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        if app.session_state is None or self._busy:
            return
        line = event.value
        event.input.value = ""
        if line.strip() == "":
            self.call_after_refresh(self._focus_prompt)
            return
        if self._handle_pending_download_answer(line):
            return
        if self._handle_models_command(line):
            return

        send_line = line
        dropped = list(self._dropped_paths)
        if dropped and paths_still_in_text(line, dropped) and not line.strip().startswith("/"):
            send_line = build_message_with_attachments(line, dropped)
            self._dropped_paths = []
        elif dropped and not paths_still_in_text(line, dropped):
            self._dropped_paths = []

        self._append_user(line if send_line == line else f"{line}  [+attached file(s)]")
        self.run_worker(lambda: self._turn_worker(send_line), thread=True, exclusive=True)

    def _download_model_worker(self, preset_key: str) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        app.call_from_thread(self._set_busy, True, f"Downloading {preset_key}...")
        progress_total = {"value": 0}
        try:
            os.environ["TQDM_POSITION"] = "-1"
            local_dir = resolve_preset_dir(preset_key, app.cwd)

            def on_hub_status(text: str) -> None:
                app.post_message(HubLogLine(f"hub: {text}"))

            def emit_progress(n: int, total: int, phase: str) -> None:
                if total > progress_total["value"]:
                    progress_total["value"] = total
                effective_total = total if total > 0 else progress_total["value"]
                status = format_download_progress(n, effective_total, phase)
                app.post_progress_status(status, n=n, total=effective_total, phase=phase)

            def on_hub_progress(n: int, total: int, label: str) -> None:
                phase, _detail = split_hub_progress_label(label)
                emit_progress(n, total, phase)

            def on_disk_bytes(nbytes: int) -> None:
                emit_progress(nbytes, progress_total["value"], "on-disk download")

            tqdm_class = hub_tqdm_bridge_factory(throttled_progress_callback(on_hub_progress))
            with watch_local_download_bytes(local_dir, on_disk_bytes):
                with capture_hub_download_logs(lambda text: app.post_message(HubLogLine(text))):
                    with capture_hub_user_warnings(
                        lambda text: app.post_message(HubLogLine(f"(warning: {text})"))
                    ):
                        path = download_preset_snapshot(
                            preset_key,
                            tqdm_class=tqdm_class,
                            verbose=True,
                            on_status=on_hub_status,
                        )
            app.call_from_thread(self._append_system, f"(downloaded {preset_key} to {path})")
        except Exception as exc:
            app.call_from_thread(self._append_system, f"(download failed: {exc})")
        finally:
            app.call_from_thread(self._set_busy, False)

    def _switch_model_worker(self, target: str) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        state = app.session_state
        if state is None:
            return
        app.call_from_thread(self._set_busy, True, f"Loading {target}...")
        try:
            switch_session_model(state, target, on_load_progress=app.progress_from_thread)
        finally:
            app.call_from_thread(self._set_busy, False)
            app.call_from_thread(self._render_chat_banner)
            app.call_from_thread(self.post_message, TurnComplete())

    def _turn_worker(self, line: str) -> None:
        app = self.app
        assert isinstance(app, OrodruinTuiApp)
        state = app.session_state
        if state is None:
            return
        app.call_from_thread(
            self._set_busy,
            True,
            "Listening…"
            if line.strip().lower().startswith("/listen")
            else "Finetuning..."
            if line.strip().lower().startswith("/finetune")
            else "Thinking…",
        )
        try:
            handled, should_generate = dispatch_chat_line(state, line)
            if state.exit_requested:
                app.call_from_thread(app.exit)
                return
            if handled and not should_generate:
                app.call_from_thread(self._render_chat_banner)
                return
            if not handled:
                state.messages.append({"role": "user", "content": line.rstrip()})
            app.call_from_thread(self._set_busy, True, "Generating…")
            run_chat_generation(state)
        finally:
            app.call_from_thread(self._set_busy, False)
            app.call_from_thread(self._render_chat_banner)
            app.call_from_thread(self.post_message, TurnComplete())

    def _activate_model_entry(self, entry: ModelPickerEntry) -> None:
        if self._busy or self.app.session_state is None or entry.kind in ModelListItem._NON_SELECTABLE:
            return
        if not entry.local:
            self._prompt_download(entry)
            return
        if entry.current:
            self._append_system(f"(already using {entry.title})")
            return
        self._append_user(f"/model {entry.target}")
        self.run_worker(lambda: self._switch_model_worker(entry.target), thread=True, exclusive=True)

    def on_turn_complete(self, _event: TurnComplete) -> None:
        self._render_chat_banner()
        self._focus_prompt()

    def on_resize(self, _event: events.Resize) -> None:
        if self.app.session_state is not None:
            self._render_chat_banner()


class OrodruinTuiApp(App):
    ENABLE_COMMAND_PALETTE = True
    TITLE = DASHBOARD_TITLE
    MODES = {
        "dashboard": DashboardScreen,
        "workshop": WorkshopScreen,
        "editor": EditorScreen,
        "chat": ChatScreen,
    }
    CSS = """
        $primary: #2563eb;
        $secondary: #1e3a8a;
        $accent: #38bdf8;
        $warning: #60a5fa;
        $success: #22d3ee;
        $error: #f87171;

        Screen {
            layout: vertical;
        }

        #nav-bar {
            dock: top;
            height: 1;
            min-height: 1;
            layout: horizontal;
            background: $surface;
            border-bottom: solid $secondary;
        }

        #nav-dashboard, #nav-workshop, #nav-editor, #nav-chat {
            min-width: 12;
            height: 1;
            min-height: 1;
            border: none;
            margin-right: 1;
        }

        #nav-spacer {
            width: 1fr;
        }

        #dash-root {
            height: 1fr;
            padding: 0 1;
        }

        #dash-grid {
            layout: grid;
            grid-size: 2;
            grid-gutter: 1 1;
            height: 1fr;
        }

        .tile-span-2 {
            column-span: 2;
        }

        #dash-notes {
            height: 1;
            color: $text-muted;
            padding: 0 1;
        }

        #home-row {
            height: 1fr;
        }

        #chat-column {
            height: 1fr;
            width: 1fr;
        }

        #home-left {
            width: 44;
            min-width: 32;
            max-width: 56;
            border: solid $secondary;
            background: $surface;
        }

        #home-project-title, #model-picker-title {
            width: 1fr;
            height: 1;
            padding: 0 1;
            background: $primary;
            color: $text;
            text-style: bold;
        }

        #home-tree {
            height: 1fr;
            overflow-x: auto;
        }

        #home-main {
            width: 1fr;
        }

        #home-stats {
            height: 3;
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

        #home-content-body {
            height: 1fr;
            border: solid $primary;
            scrollbar-gutter: stable;
        }

        #chat-banner {
            height: auto;
            min-height: 9;
            max-height: 14;
            padding: 0 1;
            background: $surface;
            border: solid $primary;
            color: $text;
        }

        #chat-toolbar {
            height: 1;
            layout: horizontal;
            background: $surface;
            border: solid $primary;
            border-top: none;
            border-bottom: none;
        }

        #chat-toolbar-title {
            width: 1fr;
            padding: 0 1;
            color: $text-muted;
        }

        #chat-copy {
            min-width: 8;
            height: 1;
            border: none;
        }

        #transcript {
            height: 1fr;
            border: solid $primary;
            scrollbar-gutter: stable;
            padding: 0 1;
            background: $background;
        }

        #transcript:focus-within {
            border: solid $accent;
        }

        RichLog {
            background: $background;
        }

        #home-log, #home-markdown {
            height: 1fr;
            border: none;
        }

        #editor-body {
            height: 1fr;
        }

        #editor-row {
            height: 1fr;
            min-height: 8;
        }

        #editor-files-show {
            width: 3;
            min-width: 3;
            height: 1fr;
            border: none;
            background: $surface;
        }

        #editor-files-col {
            width: 48;
            min-width: 16;
            border: solid $secondary;
            background: $surface;
        }

        #editor-project-menubar, #editor-source-menubar, #editor-preview-menubar {
            height: 1;
            min-height: 1;
            width: 1fr;
            background: $surface;
        }

        #editor-project-menubar Button, #editor-source-menubar Button, #editor-preview-menubar Button {
            height: 1;
            min-height: 1;
            min-width: 6;
            border: none;
            background: $surface;
            padding: 0 1;
        }

        #editor-aux-bar {
            height: 1;
            layout: horizontal;
            background: $surface;
        }

        #editor-aux-toggle {
            min-width: 8;
            height: 1;
            border: none;
        }

        #editor-aux-tab-terminal, #editor-aux-tab-chat, #editor-aux-tab-logs, #editor-aux-tab-errors {
            min-width: 10;
            height: 1;
            border: none;
            margin-right: 1;
        }

        #editor-aux-spacer {
            width: 1fr;
        }

        #editor-tree {
            height: 1fr;
            border-top: solid $primary;
            overflow-x: auto;
        }

        #editor-source-col {
            width: 1fr;
            min-width: 24;
            border: solid $primary;
            background: $background;
        }

        #editor-preview-col {
            width: 1fr;
            min-width: 20;
            border: solid $primary;
            background: $background;
        }

        #editor-source {
            height: 1fr;
            border: none;
        }

        #editor-source .text-area--suggestion {
            color: $text-muted;
        }

        #editor-preview-scroll {
            height: 1fr;
            border: none;
        }

        #editor-preview {
            height: auto;
            margin: 0 1;
            padding-bottom: 1;
        }

        #editor-raster {
            height: 1fr;
            width: 1fr;
            border: none;
        }

        #editor-table {
            height: 1fr;
            width: 1fr;
            border: none;
        }

        #editor-notebook {
            height: 1fr;
            width: 1fr;
            border: none;
        }

        #editor-aux {
            height: 10;
            min-height: 3;
            border: solid $secondary;
            background: $surface;
        }

        #editor-aux.-collapsed {
            height: 3;
            min-height: 3;
        }

        #editor-aux-switch {
            height: 1fr;
            padding: 0 1;
        }

        #editor-aux-terminal {
            height: 1fr;
            layout: vertical;
            padding: 0;
        }

        #editor-aux-terminal-cwd {
            height: 1;
            padding: 0 1;
            color: $text-muted;
        }

        #editor-aux-terminal-log {
            height: 1fr;
            border: none;
        }

        #editor-aux-terminal-prompt {
            height: 3;
            border: tall $primary;
        }

        #editor-aux-terminal-prompt:focus-within {
            border: tall $accent;
        }

        #editor-aux-chat {
            height: 1fr;
            layout: vertical;
            padding: 0;
        }

        #editor-aux-chat-status {
            height: 1;
            padding: 0 1;
            color: $text-muted;
        }

        #editor-aux-chat-log {
            height: 1fr;
            border: none;
        }

        #editor-aux-chat-prompt {
            height: 3;
            border: tall $primary;
        }

        #editor-aux-chat-prompt:focus-within {
            border: tall $accent;
        }

        #editor-aux-logs, #editor-aux-errors {
            height: 1fr;
            border: none;
        }

        #editor-unsaved-dialog {
            width: 56;
            height: auto;
            border: thick $primary;
            background: $surface;
            padding: 1 2;
        }

        #editor-unsaved-title {
            text-style: bold;
            margin-bottom: 1;
        }

        #editor-unsaved-actions {
            height: auto;
            margin-top: 1;
            align: center middle;
        }

        #editor-unsaved-actions Button {
            margin: 0 1;
        }

        #editor-path-dialog {
            width: 88;
            max-width: 94%;
            height: 34;
            max-height: 86%;
            border: thick $primary;
            background: $surface;
            padding: 1 1;
        }

        #editor-path-title, #editor-name-title {
            text-style: bold;
            height: 1;
            margin-bottom: 1;
        }

        #editor-path-nav {
            height: 1;
            layout: horizontal;
            margin-bottom: 1;
        }

        #editor-path-nav Button {
            min-width: 6;
            height: 1;
            border: none;
            margin-right: 1;
        }

        #editor-path-nav-spacer {
            width: 1fr;
        }

        #editor-path-wsl {
            height: 1;
            layout: horizontal;
            margin-bottom: 1;
        }

        #editor-path-wsl Button {
            min-width: 8;
            height: 1;
            border: none;
            margin-right: 1;
        }

        .editor-path-nav-label {
            width: auto;
            height: 1;
            content-align: left middle;
            margin-right: 1;
            color: $text-muted;
        }

        #editor-path-tree {
            height: 1fr;
            border: solid $secondary;
        }

        #editor-path-entry {
            height: 3;
            layout: horizontal;
            margin-top: 1;
        }

        #editor-path-input {
            width: 1fr;
            margin-top: 0;
        }

        #editor-name-input {
            margin-top: 1;
        }

        #editor-path-go {
            min-width: 8;
            height: 3;
            margin-left: 1;
        }

        #editor-path-actions, #editor-name-actions {
            height: auto;
            margin-top: 1;
            align: center middle;
        }

        #editor-path-actions Button, #editor-name-actions Button {
            margin: 0 1;
        }

        #editor-name-dialog {
            width: 56;
            height: auto;
            border: thick $primary;
            background: $surface;
            padding: 1 2;
        }

        #home-prompt, #prompt {
            dock: bottom;
            border: tall $primary;
        }

        #home-prompt:focus-within, #prompt:focus-within {
            border: tall $accent;
        }

        ModelPickerScreen {
            align: center middle;
            background: $background 60%;
        }

        #model-picker-dialog {
            width: 72;
            max-width: 90%;
            height: 80%;
            max-height: 36;
            border: solid $primary;
            background: $surface;
        }

        #model-picker-list {
            height: 1fr;
            border-top: solid $secondary;
            border-bottom: solid $secondary;
        }

        #model-picker-list:focus-within {
            border-top: solid $warning;
            border-bottom: solid $warning;
        }

        #progress-panel {
            height: 2;
            padding: 0 1;
            background: $surface;
            border: solid $primary;
            border-top: none;
        }

        #download-phase, #download-bar, #model-picker-hint, #status-bar {
            height: 1;
            padding: 0 1;
            color: $text-muted;
        }

        #download-phase {
            text-style: italic;
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
        self.load_progress_n = 0
        self.load_progress_total = 0
        self.load_progress_phase = ""
        self.model_status_text = "loading..."
        self._action_started_at: float | None = None
        self.cwd = orodruin_project_root()
        self._stderr_prev = sys.__stderr__
        self._pending_system_lines: list[str] = []
        self._pending_assistant_lines: list[str] = []
        self._throttled_emit_progress = throttled_progress_callback(self._emit_throttled_progress)

    async def on_mount(self) -> None:
        sys.stderr = TuiStderrSink(self.session_log)  # type: ignore[assignment]
        configure_chat_io(self._wrap_chat_io())
        initial_mode = "chat" if self.params.chat_first else "dashboard"
        self._apply_window_title(CHAT_TITLE if initial_mode == "chat" else DASHBOARD_TITLE)
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

        def on_dictate(text: str) -> None:
            self.call_from_thread(self._dictate_into_prompt, text)

        return ChatIo(emit=emit, on_assistant=on_assistant, on_dictate=on_dictate)

    def _dictate_into_prompt(self, text: str) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.insert_dictated_text(text)
            return
        if isinstance(screen, EditorScreen):
            try:
                prompt = screen.query_one("#editor-aux-chat-prompt", Input)
            except NoMatches:
                self.append_system(text)
                return
            existing = prompt.value.strip()
            body = text.strip()
            if not body:
                return
            prompt.value = f"{existing} {body}".strip() if existing else body
            prompt.cursor_position = len(prompt.value)
            screen.append_chat_system("(dictated into prompt · edit or Enter to send)")
            return
        self.append_system(text)

    def start_action_timer(self, *, reset: bool = False) -> None:
        if reset or self._action_started_at is None:
            self._action_started_at = time.monotonic()

    def stop_action_timer(self) -> None:
        self._action_started_at = None

    def format_with_elapsed(self, text: str) -> str:
        if self._action_started_at is None or not text.strip():
            return text
        elapsed = time.monotonic() - self._action_started_at
        suffix = f" · {_format_action_elapsed(elapsed)}"
        lines = text.split("\n", 1)
        lines[0] = lines[0].rstrip() + suffix
        return "\n".join(lines)

    def _preload_worker(self) -> None:
        self.call_from_thread(lambda: self.start_action_timer(reset=True))
        try:
            state = prepare_chat_session_or_shell(self.params, on_load_progress=self.progress_from_thread)
        except Exception as exc:
            self.call_from_thread(self._set_session_failed, f"load failed: {exc}")
            return
        self.call_from_thread(self._set_session_ready, state)

    def _emit_throttled_progress(self, n: int, total: int, payload: str) -> None:
        phase = ""
        if payload.strip():
            phase, _ = split_hub_progress_label(payload.replace("\n", " · "))
        self.post_progress_status(payload, n=n, total=total, phase=phase)

    def post_progress_status(
        self,
        status: str,
        *,
        n: int = 0,
        total: int = 0,
        phase: str = "",
    ) -> None:
        self.post_message(LoadProgress(status, n=n, total=total, phase=phase))

    def on_load_progress(self, event: LoadProgress) -> None:
        self.set_load_status(
            event.status,
            n=event.n,
            total=event.total,
            phase=event.phase,
        )

    def on_hub_log_line(self, event: HubLogLine) -> None:
        self.append_system(event.text)

    def progress_from_thread(self, n: int, total: int, label: str) -> None:
        if label.strip():
            phase, _ = split_hub_progress_label(label)
            payload = format_download_progress(n, total, phase)
        elif total > 0 or n > 0:
            payload = format_download_progress(n, total)
        else:
            payload = "working..."
        self._throttled_emit_progress(n, total, payload)

    def set_load_status(
        self,
        status: str,
        *,
        n: int = 0,
        total: int = 0,
        phase: str = "",
    ) -> None:
        if self.session_state is None or self._action_started_at is not None:
            self.start_action_timer()
        self.load_status = status
        self.load_progress_n = n
        self.load_progress_total = total
        self.load_progress_phase = phase or status
        self.model_status_text = self.format_with_elapsed(status)
        screen = self.screen
        try:
            if isinstance(screen, (DashboardScreen, WorkshopScreen)):
                screen.refresh_stats()
            elif isinstance(screen, ChatScreen):
                screen.update_load_status()
            elif isinstance(screen, EditorScreen):
                screen.refresh_chat_pane()
        except NoMatches:
            pass

    def _set_session_ready(self, state) -> None:
        self.stop_action_timer()
        self.session_state = state
        self.session_error = ""
        self.load_progress_n = 0
        self.load_progress_total = 0
        self.load_progress_phase = ""
        if chat_session_ready(state):
            if is_server_backend(state.backend_id):
                label = f"{state.backend_id}:{state.server_model}"
            else:
                label = state.preset_key or state.model_path
            self.load_status = f"ready ({label})"
        elif is_server_backend(state.backend_id):
            self.load_status = f"backend={state.backend_id} (pick a model with /models)"
        else:
            self.load_status = "no weights at resolved path (use /models)"
        self.model_status_text = self.load_status
        self.post_message(SessionReady())
        self.refresh_active_screen()

    def _set_session_failed(self, error: str) -> None:
        self.stop_action_timer()
        self.session_error = error
        self.model_status_text = error
        self.post_message(SessionFailed(error))
        try:
            self.refresh_active_screen()
        except NoMatches:
            pass

    def refresh_active_screen(self) -> None:
        screen = self.screen
        if isinstance(screen, (DashboardScreen, WorkshopScreen)):
            screen.refresh_stats()
        elif isinstance(screen, ChatScreen):
            screen.refresh_from_state()
        elif isinstance(screen, EditorScreen):
            screen.refresh_chat_pane()

    async def open_workshop(self) -> None:
        self._apply_window_title(WORKSHOP_TITLE)
        await self.switch_mode("workshop")
        self.call_after_refresh(self.refresh_active_screen)

    async def open_editor(self) -> None:
        self._apply_window_title(EDITOR_TITLE)
        await self.switch_mode("editor")
        self.call_after_refresh(self.refresh_active_screen)

    async def open_chat(self) -> None:
        self._apply_window_title(CHAT_TITLE)
        await self.switch_mode("chat")
        self.call_after_refresh(self.refresh_active_screen)

    async def open_dashboard(self) -> None:
        self._apply_window_title(DASHBOARD_TITLE)
        await self.switch_mode("dashboard")
        self.call_after_refresh(self.refresh_active_screen)

    async def cycle_mode(self) -> None:
        current = getattr(self, "current_mode", None) or "dashboard"
        if current not in MODE_CYCLE:
            current = "dashboard"
        nxt = MODE_CYCLE[(MODE_CYCLE.index(current) + 1) % len(MODE_CYCLE)]
        if nxt == "dashboard":
            await self.open_dashboard()
        elif nxt == "workshop":
            await self.open_workshop()
        elif nxt == "editor":
            await self.open_editor()
        else:
            await self.open_chat()

    def append_system(self, text: str) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.append_system_from_app(text)
        elif isinstance(screen, EditorScreen):
            screen.append_chat_system(text)
        else:
            self._pending_system_lines.append(text)

    def append_assistant(self, text: str) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.append_assistant_from_app(text)
        elif isinstance(screen, EditorScreen):
            screen.append_chat_assistant(text)
        else:
            self._pending_assistant_lines.append(text)

    def run_shell_command(self, command: str) -> None:
        from integrations.shell.runner import ShellSession

        session = ShellSession(cwd=Path(self.cwd).resolve())
        result = session.run(command)
        self.cwd = session.cwd
        if result.handled_as_cd:
            self.call_from_thread(self._append_home_log, f"[dim]cwd -> {self.cwd}[/dim]")
            return
        if result.output.strip():
            self.call_from_thread(self._append_home_log, result.output.rstrip())
        self.call_from_thread(self._append_home_log, f"[dim](exit {result.exit_code})[/dim]")

    def _handle_cd(self, command: str) -> bool:
        from integrations.shell.runner import ShellSession

        session = ShellSession(cwd=Path(self.cwd).resolve())
        result = session.handle_cd_command(command)
        if result is None:
            return False
        self.cwd = session.cwd
        if result.exit_code != 0:
            self.call_from_thread(self._append_home_log, f"[red]{result.output}[/red]")
        return True

    def _append_home_log(self, text: str) -> None:
        screen = self.screen
        if isinstance(screen, WorkshopScreen):
            screen.append_log(text)
        elif isinstance(screen, EditorScreen):
            screen.append_terminal_log(text)


def run_tui_app(params: ChatCliParams, session_log: TuiSessionLog | None) -> None:
    OrodruinTuiApp(params, session_log).run()
