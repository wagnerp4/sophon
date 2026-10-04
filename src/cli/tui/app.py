from __future__ import annotations

import datetime as _dt
import os
import re
import sys
import threading
import time
from concurrent.futures import Future
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
    Input,
    Label,
    ListItem,
    ListView,
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
    play_speech_clip,
    reset_chat_io,
    run_chat_generation,
    sst_recording,
    switch_session_model,
    transcribe_audio_paths,
)
from cli.chat_display import (
    SESSION_SEPARATOR,
    chat_prompt_placeholder,
    chat_session_header_lines,
    format_speech_clip_line,
    nexus_wordmark_markup,
    format_turn_trace_lines,
    format_user_turn_line,
    unpack_assistant_pending,
)
from cli.host.session_log import TuiSessionLog
from cli.tui.footer import NexusFooter
from cli.tui.io import TuiStderrSink, copy_text_to_system_clipboard
from cli.tui.paste_drop import (
    build_message_with_attachments,
    describe_dropped_paths,
    extract_dropped_paths,
    format_paths_for_prompt,
    paths_still_in_text,
    split_audio_paths,
)
from cli.tui.screens.dashboard import DashboardScreen
from cli.tui.screens.editor import EditorScreen
from cli.tui.screens.nav import (
    CHAT_TITLE,
    DASHBOARD_TITLE,
    EDITOR_TITLE,
    MODE_CYCLE,
    NAV_BINDINGS,
    ModeNavigationMixin,
)
from cli.tui.chat_prompt import FilesDropped
from cli.tui.keybinds import CHAT_KEYBIND_ACTIONS, SlashDispatchMixin, apply_keybinds, load_keybinds
from cli.path_highlights import ensure_highlights_file
from cli.tui.permission import PermissionPrompt
from cli.tui.choice import ChoicePrompt
from cli.tui.setup_prompt import SETUP_SKIP, SetupPrompt
from cli.tui.speech import ChatLogPane, ChatTranscript, PushToTalkMixin, TranscriptHostMixin
from utils.device.env_bootstrap import sophon_project_root
from utils.download.hf import (
    capture_hub_download_logs,
    capture_hub_user_warnings,
    download_preset_snapshot,
    format_download_bar_line,
    format_download_progress,
    hub_tqdm_bridge_factory,
    is_file_count_progress,
    normalize_hub_phase,
    parse_hub_size_hint,
    scan_local_dir_download_bytes,
    split_hub_progress_label,
    throttled_progress_callback,
    watch_local_download_bytes,
)


HUB_BYTE_TOTAL_MIN = 4097
_MODELS_CMD = re.compile(r"^/models(?:\s+(all|local|missing))?\s*$", re.IGNORECASE)


def _set_console_title(title: str) -> None:
    if sys.platform == "win32" and os.environ.get("SOPHON_TUI_CHILD"):
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


_ELAPSED_SUFFIX_RE = re.compile(r" · (?:\d+s|\d+m \d{2}s|\d+h \d+m)$")


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


class LoadProgress(Message):
    def __init__(
        self,
        status: str,
        *,
        n: int = 0,
        total: int = 0,
        phase: str = "",
        bytes_per_s: float = 0.0,
    ) -> None:
        self.status = status
        self.n = n
        self.total = total
        self.phase = phase
        self.bytes_per_s = bytes_per_s
        super().__init__()


class HubLogLine(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class ModelListItem(ListItem):
    _NON_SELECTABLE = frozenset({"group", "subgroup"})

    def __init__(self, entry: ModelPickerEntry, *, expanded: bool = False) -> None:
        self.entry = entry
        if entry.kind == "header":
            arrow = "▾" if expanded and entry.child_count > 0 else "▸"
            extra = f"  {entry.detail}" if entry.detail else ""
            super().__init__(Label(f"[bold]{arrow} {entry.title}{extra}[/bold]"))
            return
        if entry.kind == "group":
            super().__init__(Label(f"[bold dim]{entry.title}[/bold dim]"), disabled=True)
            return
        if entry.kind == "subgroup":
            super().__init__(Label(f"[dim]  {entry.title}[/dim]"), disabled=True)
            return
        if entry.local:
            marker = "  ▸ " if entry.current else "    "
        else:
            marker = "      ▸ " if entry.current else "      "
        suffix = "" if entry.local else " [dim]remote[/dim]"
        super().__init__(Label(f"{marker}{entry.title}{suffix}"))

    def on_click(self, event: events.Click) -> None:
        if event.chain >= 2 and self.entry.kind == "header":
            self.post_message(ModelItemDoubleClicked(self.entry))
            event.stop()
            return
        if event.chain >= 2 and self.entry.kind not in self._NON_SELECTABLE:
            self.post_message(ModelItemDoubleClicked(self.entry))
            event.stop()


class ModelPickerScreen(ModalScreen[ModelPickerEntry | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Close", show=True),
        Binding("ctrl+m", "cancel", "Close", show=False),
        Binding("left", "collapse_group", "Collapse", show=False),
        Binding("right", "expand_group", "Expand", show=False),
    ]

    def __init__(self, entries: list[ModelPickerEntry]) -> None:
        super().__init__()
        self._entries = entries
        self._expanded = {entry.group for entry in entries if entry.current and entry.group}

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-dialog"):
            yield Static(
                "Models — ↑↓ move · Enter expand/load · ←/→ groups · Esc close",
                id="model-picker-title",
            )
            yield ListView(id="model-picker-list")
            yield Static("Enter on a group to expand. Enter on a model to load.", id="model-picker-hint")

    def on_mount(self) -> None:
        self._fill_list()

    def _visible_entries(self) -> list[ModelPickerEntry]:
        visible: list[ModelPickerEntry] = []
        for entry in self._entries:
            if entry.kind == "header":
                visible.append(entry)
                continue
            if entry.group and entry.group in self._expanded:
                visible.append(entry)
        return visible

    def _fill_list(self, *, focus_group: str | None = None) -> None:
        panel = self.query_one("#model-picker-list", ListView)
        panel.clear()
        visible = self._visible_entries()
        if not visible:
            panel.append(ListItem(Label("[dim]no models[/dim]"), disabled=True))
            return
        highlight_index = 0
        for index, entry in enumerate(visible):
            opened = entry.group in self._expanded
            panel.append(ModelListItem(entry, expanded=opened))
            if focus_group and entry.kind == "header" and entry.group == focus_group:
                highlight_index = index
            elif entry.current and entry.group in self._expanded:
                highlight_index = index
        panel.index = highlight_index
        panel.focus()

    def _highlighted_entry(self) -> ModelPickerEntry | None:
        panel = self.query_one("#model-picker-list", ListView)
        index = panel.index
        visible = self._visible_entries()
        if index is None or index < 0 or index >= len(visible):
            return None
        return visible[index]

    def _toggle_group(self, group: str, child_count: int) -> None:
        if not group or child_count <= 0:
            return
        if group in self._expanded:
            self._expanded.discard(group)
        else:
            self._expanded.add(group)
        self._fill_list(focus_group=group)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_collapse_group(self) -> None:
        entry = self._highlighted_entry()
        if entry is None or not entry.group:
            return
        self._expanded.discard(entry.group)
        self._fill_list(focus_group=entry.group)

    def action_expand_group(self) -> None:
        entry = self._highlighted_entry()
        if entry is None or not entry.group:
            return
        self._expanded.add(entry.group)
        self._fill_list(focus_group=entry.group)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if not isinstance(item, ModelListItem):
            return
        if item.entry.kind == "header":
            self._toggle_group(item.entry.group, item.entry.child_count)
            return
        if item.entry.kind not in ModelListItem._NON_SELECTABLE:
            self.dismiss(item.entry)

    def on_model_item_double_clicked(self, event: ModelItemDoubleClicked) -> None:
        if event.entry.kind == "header":
            self._toggle_group(event.entry.group, event.entry.child_count)
            return
        if event.entry.kind not in ModelListItem._NON_SELECTABLE:
            self.dismiss(event.entry)


class ChatScreen(TranscriptHostMixin, SlashDispatchMixin, PushToTalkMixin, ModeNavigationMixin, Screen):
    BINDINGS = [
        Binding("f1", "show_help", "Help", show=True),
        Binding("ctrl+p", "command_palette", "Palette", show=True),
        Binding("ctrl+m", "open_models", "Models", show=True),
        *NAV_BINDINGS,
        Binding("ctrl+y", "copy_chat", "Copy", show=False),
        Binding("ctrl+l", "listen", "Speak", show=False, priority=True),
        Binding("ctrl+shift+t", "focus_transcript", "Log", show=True),
        Binding("tab", "cycle_focus", "Focus", show=False),
        Binding("escape", "focus_prompt", "Input", show=True),
        Binding("ctrl+c", "absorb_ctrl_c", show=False, priority=True),
        Binding("ctrl+q", "request_quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._init_transcript_host()
        self._focus_target = "prompt"
        self._started = False
        self._busy = False
        self._pending_download: ModelPickerEntry | None = None
        self._session_started_at = _dt.datetime.now()
        self._picker_open = False
        self._activity_kind = ""
        self._dropped_paths: list[Path] = []
        self._mic_hold_started = False
        self._mic_transcribing = False

    def _query_widget(self, selector: str, widget_type: type):
        try:
            return self.query_one(selector, widget_type)
        except NoMatches:
            return None

    def _update_status_bar(self, text: str) -> None:
        status = self._query_widget("#status-bar", Static)
        if status is None:
            return
        body = text if text else ""
        status.update(body)
        status.display = bool(body.strip())

    def compose(self) -> ComposeResult:
        with Vertical(id="chat-column"):
            yield Static(nexus_wordmark_markup(), id="chat-toolbar-title", markup=True)
            yield ChatLogPane(transcript_id="transcript", prompt_id="prompt")
            with Vertical(id="progress-panel"):
                yield Static("", id="download-phase")
                yield Static("", id="download-bar")
                yield Static("", id="train-curve")
            yield Static("", id="status-bar")
        yield NexusFooter(chat_actions=True)

    def on_mount(self) -> None:
        panel = self._query_widget("#progress-panel", Vertical)
        if panel is not None:
            panel.display = False
        self.refresh_from_state()
        ensure_highlights_file(sophon_project_root())
        self._apply_chat_keybinds()
        self.set_interval(0.25, self._tick_action_elapsed)

    def on_screen_resume(self) -> None:
        self._apply_chat_keybinds()

    def _apply_chat_keybinds(self) -> None:
        apply_keybinds(
            self,
            load_keybinds(sophon_project_root()),
            actions=CHAT_KEYBIND_ACTIONS,
        )
        from cli.tui.keybinds import refresh_keybind_surfaces

        refresh_keybind_surfaces(self.app)

    def _tick_action_elapsed(self) -> None:
        app = self.app
        if not isinstance(app, SophonTuiApp):
            return
        if self._busy:
            if app.load_progress_total > 0 or app.load_progress_n > 0:
                self.update_load_status()
            else:
                self._show_activity_panel()
            self._pulse_heartbeat()
            return
        if self.sst_recording():
            self._update_status_bar(self._status_text())
            return
        if app._action_started_at is None:
            return
        if app.session_state is None:
            self.update_load_status()

    def refresh_from_state(self) -> None:
        app = self.app
        assert isinstance(app, SophonTuiApp)
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
        if not self._started:
            for line in chat_session_header_lines(app.session_state, for_markup=True):
                if line.plain:
                    self._write_transcript(line)
            self._append_system(SESSION_SEPARATOR)
            stamp = self._session_started_at.strftime("%Y-%m-%d %H:%M:%S")
            self._append_system(f"session started {stamp}")
            self._started = True
        pending_sys = app._pending_system_lines
        if pending_sys:
            for text in list(pending_sys):
                self._append_system(text)
            pending_sys.clear()
        pending_asst = app._pending_assistant_lines
        if pending_asst:
            for item in list(pending_asst):
                text, trace = unpack_assistant_pending(item)
                self._append_assistant(text, trace)
            pending_asst.clear()
        pending_clips = app._pending_speech_clips
        if pending_clips:
            for clip in list(pending_clips):
                self.append_speech_clip(clip)
            pending_clips.clear()
        self._hide_progress_panel()
        prompt = self._query_widget("#prompt", Input)
        if prompt is not None:
            prompt.disabled = False
            prompt.placeholder = chat_prompt_placeholder(app.session_state)
        self._update_status_bar(self._status_text())
        self.call_after_refresh(self._focus_prompt)

    def update_load_status(self) -> None:
        app = self.app
        assert isinstance(app, SophonTuiApp)
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
            panel.styles.height = 2
        curve = self._query_widget("#train-curve", Static)
        if curve is not None:
            curve.update("")
            curve.display = False

    def _activity_label(self) -> str:
        app = self.app
        assert isinstance(app, SophonTuiApp)
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
        app = self.app
        if "download" in kind:
            if isinstance(app, SophonTuiApp) and (
                app.load_progress_total > 0 or app.load_progress_n > 0
            ):
                self._update_progress_panel(
                    app.load_progress_n,
                    app.load_progress_total,
                    app.load_progress_phase or "downloading",
                )
                return
            hint = format_download_bar_line(0, 0)
        elif "load" in kind:
            hint = "loading weights / switching model"
        elif "finetun" in kind:
            hint = "LoRA SFT · steps and loss stream in chat"
        else:
            hint = "in progress · server calls are not cancelled by Esc"
        bar_widget.update(hint)
        curve_widget = self._query_widget("#train-curve", Static)
        if curve_widget is not None:
            losses = []
            if isinstance(app, SophonTuiApp) and app.session_state is not None:
                losses = list(getattr(app.session_state, "train_loss_history", []) or [])
            if losses and "finetun" in kind:
                from training.common.curves import sparkline

                curve_widget.update(f"loss {sparkline(losses)} {losses[-1]:.4f}")
                curve_widget.display = True
                panel.styles.height = 3
            else:
                curve_widget.update("")
                curve_widget.display = False
                panel.styles.height = 2
        status = self._query_widget("#status-bar", Static)
        if status is not None:
            self._update_status_bar("")

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
        if isinstance(app, SophonTuiApp):
            clean_phase = app.format_with_elapsed(clean_phase)
        phase_widget.update(clean_phase)
        bps = float(getattr(app, "load_progress_bps", 0.0) or 0.0) if isinstance(app, SophonTuiApp) else 0.0
        if is_file_count_progress(n, total, phase) or is_file_count_progress(n, total, clean_phase):
            bar_widget.update(format_download_bar_line(n, total, file_count=True))
        else:
            bar_widget.update(format_download_bar_line(n, total, bytes_per_s=bps))
        status = self._query_widget("#status-bar", Static)
        if status is not None:
            self._update_status_bar("")

    def _status_text(self) -> str:
        app = self.app
        assert isinstance(app, SophonTuiApp)
        if self.sst_recording():
            rec = getattr(self.app.session_state, "sst_mic", None)
            elapsed = float(getattr(rec, "elapsed_s", 0.0) or 0.0)
            return f"Recording {elapsed:.0f}s · Stop or Ctrl+L to finish · Esc cancel"
        if self._busy:
            return self._activity_label()
        state = app.session_state
        if state is not None and bool(getattr(state, "ssl4sed_running", False)):
            target = str(getattr(state, "trainer_target", "") or getattr(state, "ssl4sed_default_target", "") or "ssl4sed")
            return f"ssl4sed running {target}  /trainer watch | /trainer stop"
        if app.session_state is None:
            return f"Loading model... {app.load_status}"
        return ""

    def _set_busy(self, active: bool, status: str = "") -> None:
        self._busy = active
        app = self.app
        assert isinstance(app, SophonTuiApp)
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

    def sst_input_blocked(self) -> bool:
        return self._busy or self._mic_transcribing

    def action_focus_prompt(self) -> None:
        if self.sst_recording():
            self._cancel_sst_mic()
            return
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
        transcript = self._query_widget("#transcript", ChatTranscript)
        if transcript is not None:
            transcript.focus()
        self._update_status_bar(self._status_text())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.handle_nav_button(event.button.id):
            return
        if event.button.id == "chat-copy":
            self.action_copy_chat()

    def action_absorb_ctrl_c(self) -> None:
        self.notify("Ctrl+Y or the Copy button copies chat text. /quit or Ctrl+Q exits.", timeout=4)

    def action_request_quit(self) -> None:
        self.app.exit()

    def action_show_help(self) -> None:
        from cli.tui.keybinds import load_keybinds

        binds = load_keybinds(sophon_project_root())
        actions = binds.actions
        self._append_system("Key bindings")
        rows = (
            ("f1", "Show slash commands and key bindings"),
            (actions.get("open_models") or "ctrl+m", "Open model picker (/models)"),
            (actions.get("open_dashboard") or "ctrl+d", "Open dashboard"),
            (actions.get("open_editor") or "ctrl+e", "Open editor"),
            (actions.get("open_chat") or "ctrl+g", "Open nexus"),
            (actions.get("accept_completion") or "right", "Editor: accept autocomplete suggestion"),
            (actions.get("toggle_project") or "ctrl+b", "Editor: toggle project pane"),
            (actions.get("toggle_aux") or "ctrl+j", "Editor: toggle lower panel"),
            (actions.get("cycle_mode") or "ctrl+h", "Cycle dashboard / editor / chat"),
            (actions.get("cycle_tree") or "ctrl+shift+t", "Cycle editor tree"),
            (actions.get("listen") or "ctrl+l", "Speak: click to start, click Stop to finish"),
            (actions.get("copy_chat") or "ctrl+y", "Copy full chat log to clipboard"),
            ("copy", "Footer Copy button"),
            ("ctrl+q", "Exit the TUI (/quit also works)"),
            ("tab", "Switch focus between chat log and input"),
            ("escape", "Return focus to the message input"),
            (actions.get("permission_once") or "1", "Harness permission: allow this time"),
            (actions.get("permission_persist") or "2", "Harness permission: allow-list"),
            (actions.get("permission_deny") or "3", "Harness permission: decline"),
        )
        for key, description in rows:
            shown = str(key or "").strip()
            if not shown:
                continue
            self._append_system(f"  {shown:<16} {description}")
        self._append_system("Slash commands")
        for line in format_chat_help().splitlines():
            self._append_system(line)
        self._focus_prompt()

    def _append_user(self, text: str) -> None:
        app = self.app
        assert isinstance(app, SophonTuiApp)
        if app.session_log is not None:
            app.session_log.write("user", text)
        self._turn_counter += 1
        self._write_transcript(
            format_user_turn_line(text, turn=self._turn_counter, for_markup=True)
        )

    def append_system_from_app(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            self._append_system(line)

    def append_assistant_from_app(self, text: str, trace: object | None = None) -> None:
        self._append_assistant(text, trace)

    def _model_picker_entries(self, app: "SophonTuiApp") -> list[ModelPickerEntry]:
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
        assert isinstance(app, SophonTuiApp)
        if self._picker_open:
            return
        self._picker_open = True

        def on_close(entry: ModelPickerEntry | None) -> None:
            self._picker_open = False
            if entry is not None:
                self._activate_model_entry(entry)
            self.call_after_refresh(self._focus_prompt)

        try:
            entries = self._model_picker_entries(app)
        except Exception as exc:
            self._picker_open = False
            self._append_system(f"(model picker failed: {exc})")
            return
        self.app.push_screen(ModelPickerScreen(entries), on_close)

    def _handle_models_command(self, line: str) -> bool:
        if not _MODELS_CMD.match(line.strip()):
            return False
        self._append_user(line)
        self._append_system("(model picker — ↑↓ / click · Enter · Esc)")
        self._open_model_picker()
        return True

    def _handle_jobs_queue(self, line: str) -> bool:
        if line.strip() not in ("/jobs", "/jobs queue"):
            return False
        from cli.tui.screens.jobs_queue import JobQueueScreen

        self._append_user(line)
        self.app.push_screen(JobQueueScreen())
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
        assert isinstance(app, SophonTuiApp)
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
        assert isinstance(app, SophonTuiApp)
        if event.input.id != "prompt":
            return
        if app.session_state is None or self._busy:
            return
        line = event.value
        event.input.value = ""
        try:
            self._submit_prompt_line(line)
        except Exception as exc:
            self._append_system(f"(send failed: {exc})")
            self.call_after_refresh(self._focus_prompt)

    def _submit_prompt_line(self, line: str) -> None:
        app = self.app
        assert isinstance(app, SophonTuiApp)
        if line.strip() == "":
            if self.sst_recording():
                self._stop_sst_mic()
            self.call_after_refresh(self._focus_prompt)
            return
        if self._handle_pending_download_answer(line):
            return
        if self._handle_models_command(line):
            return
        if self._handle_jobs_queue(line):
            return
        if self.sst_recording() and not line.strip().lower().startswith("/listen"):
            self._cancel_sst_mic()

        send_line = line
        dropped = list(self._dropped_paths)
        state = app.session_state
        from cli.key_prompt import take_pending_secret

        secret_note = take_pending_secret(state, line)
        if secret_note is not None:
            self._append_system(secret_note)
            self.call_after_refresh(self._focus_prompt)
            return
        if dropped and paths_still_in_text(line, dropped) and not line.strip().startswith("/") and line.strip() != "!":
            if bool(getattr(state, "auto_attach", True)):
                send_line = build_message_with_attachments(line, dropped)
            self._dropped_paths = []
        elif dropped:
            self._dropped_paths = []

        self._append_user(line if send_line == line else f"{line}  [+attached file(s)]")
        self.run_worker(lambda: self._turn_worker(send_line), thread=True, exclusive=True)

    def _download_model_worker(self, preset_key: str) -> None:
        app = self.app
        assert isinstance(app, SophonTuiApp)
        app.call_from_thread(self._set_busy, True, f"Downloading {preset_key}...")
        progress_total = {"value": 0}
        progress_n = {"value": 0}
        rate_state = {"t": time.monotonic(), "n": 0, "bps": 0.0, "grew_at": time.monotonic()}
        try:
            os.environ["TQDM_POSITION"] = "-1"
            local_dir = resolve_preset_dir(preset_key, app.cwd)

            def on_hub_status(text: str) -> None:
                app.post_message(HubLogLine(f"hub: {text}"))

            def emit_progress(n: int, total: int, phase: str) -> None:
                if n > progress_n["value"]:
                    progress_n["value"] = n
                shown_n = progress_n["value"]
                shown_total = total if total >= HUB_BYTE_TOTAL_MIN else progress_total["value"]
                if shown_total > progress_total["value"]:
                    progress_total["value"] = shown_total
                shown_total = progress_total["value"]
                now = time.monotonic()
                dt = now - rate_state["t"]
                dn = shown_n - rate_state["n"]
                if dn > 0:
                    rate_state["grew_at"] = now
                    if dt >= 0.4:
                        inst = dn / dt
                        prev = rate_state["bps"]
                        rate_state["bps"] = inst if prev <= 0 else (0.65 * prev + 0.35 * inst)
                        rate_state["t"] = now
                        rate_state["n"] = shown_n
                elif now - rate_state["grew_at"] > 8.0:
                    rate_state["bps"] = 0.0
                bps = float(rate_state["bps"])
                status = format_download_progress(
                    shown_n, shown_total, phase, bytes_per_s=bps
                )
                app.post_progress_status(
                    status,
                    n=shown_n,
                    total=shown_total,
                    phase=phase,
                    bytes_per_s=bps,
                )

            def on_expected_bytes(nbytes: int) -> None:
                if nbytes > progress_total["value"]:
                    progress_total["value"] = nbytes
                emit_progress(
                    scan_local_dir_download_bytes(local_dir),
                    progress_total["value"],
                    "Hub size known",
                )

            def on_hub_progress(n: int, total: int, label: str) -> None:
                phase, _detail = split_hub_progress_label(label)
                if is_file_count_progress(n, total, phase) or is_file_count_progress(n, total, label):
                    app.post_progress_status(label, n=n, total=total, phase=phase)
                    return
                if total >= HUB_BYTE_TOTAL_MIN and total > progress_total["value"]:
                    progress_total["value"] = total
                emit_progress(n, progress_total["value"], phase)

            def on_disk_bytes(nbytes: int) -> None:
                emit_progress(nbytes, progress_total["value"], "on-disk download")

            def on_hub_log(text: str) -> None:
                hinted = parse_hub_size_hint(text)
                if hinted > progress_total["value"]:
                    progress_total["value"] = hinted
                    emit_progress(progress_n["value"], hinted, "Hub size hint")
                app.post_message(HubLogLine(text))

            emit_progress(
                scan_local_dir_download_bytes(local_dir),
                0,
                f"Downloading {preset_key}...",
            )
            tqdm_class = hub_tqdm_bridge_factory(throttled_progress_callback(on_hub_progress))
            with watch_local_download_bytes(local_dir, on_disk_bytes):
                with capture_hub_download_logs(on_hub_log):
                    with capture_hub_user_warnings(
                        lambda text: app.post_message(HubLogLine(f"(warning: {text})"))
                    ):
                        path = download_preset_snapshot(
                            preset_key,
                            tqdm_class=tqdm_class,
                            verbose=True,
                            on_status=on_hub_status,
                            on_expected_bytes=on_expected_bytes,
                        )
            app.call_from_thread(self._append_system, f"(downloaded {preset_key} to {path})")
        except Exception as exc:
            app.call_from_thread(self._append_system, f"(download failed: {exc})")
        finally:
            app.call_from_thread(self._set_busy, False)

    def _switch_model_worker(self, target: str) -> None:
        app = self.app
        assert isinstance(app, SophonTuiApp)
        state = app.session_state
        if state is None:
            return
        app.call_from_thread(self._set_busy, True, f"Loading {target}...")
        try:
            switch_session_model(state, target, on_load_progress=app.progress_from_thread)
        finally:
            app.call_from_thread(self._set_busy, False)
            app.call_from_thread(self.post_message, TurnComplete())

    def _turn_busy_label(self, raw: str) -> str:
        if raw.startswith("/listen"):
            return "Transcribing…"
        if raw.startswith("/finetune"):
            return "Finetuning..."
        if raw.startswith("/trainer run"):
            state = self.app.session_state
            driver = str(getattr(state, "trainer_driver", "") or "") if state is not None else ""
            if driver == "ssl4sed" or "target=" in raw:
                return "Starting ssl4sed..."
            return "Finetuning..."
        if raw.startswith("/trainer data get"):
            return "Trainer data..."
        if raw.startswith("/trainer"):
            return "Trainer..."
        if raw.startswith("/models-sync") or raw.startswith("/models-refresh"):
            return "Syncing model index..."
        if raw.startswith("/eval") or raw.startswith("/rag-index"):
            return "Evaluating..."
        return "Thinking…"

    def _turn_worker(self, line: str) -> None:
        app = self.app
        assert isinstance(app, SophonTuiApp)
        state = app.session_state
        if state is None:
            return
        raw = line.strip().lower()
        listen_start = raw.startswith("/listen") and not sst_recording(state) and "cancel" not in raw
        if not listen_start:
            app.call_from_thread(
                self._set_busy,
                True,
                self._turn_busy_label(raw),
            )
        try:
            handled, should_generate = dispatch_chat_line(state, line)
            if state.exit_requested:
                app.call_from_thread(app.exit)
                return
            if handled and not should_generate:
                app.call_from_thread(self.refresh_from_state)
                return
            if not handled:
                from cli.chat import prepare_user_message_text

                state.messages.append(
                    {"role": "user", "content": prepare_user_message_text(state, line)}
                )
            app.call_from_thread(self._set_busy, True, "Generating…")
            run_chat_generation(state)
        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            if app.session_log is not None:
                app.session_log.write("err", tb)
            app.call_from_thread(self._append_system, f"(turn failed: {exc})")
        finally:
            app.call_from_thread(self._set_busy, False)
            app.call_from_thread(self.post_message, TurnComplete())

    def _activate_model_entry(self, entry: ModelPickerEntry) -> None:
        if self._busy or self.app.session_state is None or entry.kind in ModelListItem._NON_SELECTABLE:
            return
        if entry.kind == "adapter":
            if entry.current:
                self._append_system(f"(already using adapter {entry.title})")
                return
            self._append_user(f"/adapter load {entry.target}")
            self.run_worker(
                lambda: self._adapter_load_worker(entry.target),
                thread=True,
                exclusive=True,
            )
            return
        if not entry.local:
            self._prompt_download(entry)
            return
        if entry.current:
            self._append_system(f"(already using {entry.title})")
            return
        self._append_user(f"/model {entry.target}")
        self.run_worker(lambda: self._switch_model_worker(entry.target), thread=True, exclusive=True)

    def _adapter_load_worker(self, name: str) -> None:
        app = self.app
        assert isinstance(app, SophonTuiApp)
        state = app.session_state
        if state is None:
            return
        app.call_from_thread(self._set_busy, True, f"Loading adapter {name}...")
        try:
            dispatch_chat_line(state, f"/adapter load {name}")
        finally:
            app.call_from_thread(self._set_busy, False)
            app.call_from_thread(self.post_message, TurnComplete())

    def on_turn_complete(self, _event: TurnComplete) -> None:
        self._focus_prompt()


class SophonTuiApp(App):
    ENABLE_COMMAND_PALETTE = True
    TITLE = DASHBOARD_TITLE
    MODES = {
        "dashboard": DashboardScreen,
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
            padding: 0;
        }

        ChatScreen {
            padding: 0;
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

        #nav-dashboard, #nav-editor, #nav-chat {
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

        DashboardScreen.-dash-scroll #dash-root,
        DashboardScreen.-dash-narrow #dash-root {
            overflow-y: auto;
        }

        DashboardScreen.-dash-scroll #dash-grid,
        DashboardScreen.-dash-narrow #dash-grid {
            height: auto;
        }

        DashboardScreen.-dash-scroll #dash-grid > *,
        DashboardScreen.-dash-narrow #dash-grid > * {
            height: auto;
            min-height: 4;
        }

        DashboardScreen.-dash-narrow #dash-grid {
            grid-size: 1;
        }

        DashboardScreen.-dash-narrow .tile-span-2 {
            column-span: 1;
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
            border: solid $primary;
            background: $background;
        }

        #chat-column:focus-within {
            border: solid $accent;
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

        #chat-toolbar-title {
            width: 100%;
            height: auto;
            min-height: 8;
            padding: 0 1;
            background: $background;
            color: $text;
        }

        .chat-log-wrap {
            height: 1fr;
            layout: vertical;
            background: $background;
        }

        .slash-palette {
            width: 100%;
            height: auto;
            max-height: 12;
            display: none;
            border-top: solid $secondary;
            background: $surface;
            padding: 0;
        }

        #chat-column > .chat-log-wrap {
            border: none;
        }

        #chat-column > .chat-log-wrap:focus-within {
            border: none;
        }

        #transcript {
            height: 1fr;
            border: none;
            scrollbar-gutter: stable;
            padding: 0 1;
            background: $background;
        }

        #transcript:focus-within {
            border: none;
        }

        ChatTranscript {
            background: $background;
        }

        .transcript-line {
            width: 100%;
            height: auto;
            padding: 0;
        }

        .turn-heartbeat, .heartbeat-kind, .heartbeat-stack {
            width: 100%;
            height: auto;
            min-height: 1;
            padding: 0;
            background: $background;
        }

        .heartbeat-title, .heartbeat-kind-title {
            width: 100%;
            height: 1;
            min-height: 1;
            padding: 0 1;
            color: $text-muted;
        }

        .heartbeat-reasoning {
            width: 100%;
            height: auto;
            padding: 0 1 1 1;
            color: $text-muted;
        }

        .heartbeat-meta {
            width: 100%;
            height: auto;
            padding: 0 1;
            color: $text-muted;
        }

        .heartbeat-body {
            width: 100%;
            height: auto;
            padding: 0 1;
            color: $text-muted;
        }

        .speech-clip-bar {
            height: 1;
            width: 100%;
            layout: horizontal;
            background: $surface;
        }

        .speech-clip-label {
            width: 1fr;
            height: 1;
            padding: 0 1;
            color: $text-muted;
        }

        .speech-clip-btn {
            min-width: 6;
            height: 1;
            border: none;
            padding: 0 1;
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

        #editor-aux-tab-terminal, #editor-aux-tab-chat, #editor-aux-tab-review, #editor-aux-tab-logs, #editor-aux-tab-errors {
            min-width: 10;
            height: 1;
            border: none;
            margin-right: 1;
        }

        #editor-aux-spacer {
            width: 1fr;
        }

        #editor-tree-switch {
            height: 1fr;
        }

        #editor-tree, #editor-zotero-tree, #editor-drive-tree, #editor-bookmarks-tree, #editor-overleaf-tree {
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
            height: 14;
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

        #editor-aux-chat-status-row {
            height: 1;
            layout: horizontal;
            background: $surface;
        }

        #editor-aux-chat-status {
            width: 1fr;
            height: 1;
            padding: 0 1;
            color: $text-muted;
        }

        #editor-aux-chat-copy, #editor-aux-speak {
            min-width: 8;
            height: 1;
            border: none;
        }

        #editor-aux-chat-log {
            height: 1fr;
            border: none;
            background: $background;
            scrollbar-gutter: stable;
            padding: 0 1;
        }

        #editor-aux-chat > .chat-log-wrap {
            height: 1fr;
            border: none;
        }

        #editor-aux-review {
            height: 1fr;
            layout: vertical;
            padding: 0;
        }

        #editor-aux-review-status {
            height: 1;
            padding: 0 1;
            color: $text-muted;
        }

        #editor-aux-review-actions {
            height: 1;
            layout: horizontal;
            margin-bottom: 1;
        }

        #editor-aux-review-actions Button {
            min-width: 12;
            height: 1;
            border: none;
            margin-right: 1;
        }

        #editor-aux-review-body {
            height: 1fr;
            layout: horizontal;
        }

        #editor-aux-review-files {
            width: 36;
            height: 1fr;
            border: solid $secondary;
        }

        #editor-aux-review-diff {
            width: 1fr;
            height: 1fr;
            border: solid $secondary;
            margin-left: 1;
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

        PermissionPrompt {
            align: center middle;
            background: $background 60%;
        }

        #permission-dialog {
            width: 72;
            max-width: 90%;
            height: auto;
            max-height: 90%;
            border: thick $primary;
            background: $surface;
            padding: 1 2;
        }

        #permission-title {
            text-style: bold;
            margin-bottom: 1;
        }

        #permission-body {
            height: auto;
        }

        #permission-actions {
            height: auto;
            margin-top: 1;
            align: center middle;
        }

        #permission-actions Button {
            margin: 0 1;
        }

        SetupPrompt {
            align: center middle;
            background: $background 60%;
        }

        #setup-dialog {
            width: 72;
            max-width: 90%;
            height: auto;
            max-height: 90%;
            border: thick $primary;
            background: $surface;
            padding: 1 2;
        }

        #setup-title {
            text-style: bold;
            margin-bottom: 1;
        }

        #setup-body {
            height: auto;
            max-height: 24;
            overflow-y: auto;
        }

        #setup-actions {
            height: auto;
            margin-top: 1;
            align: center middle;
        }

        #setup-actions Button {
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

        .transcript-prompt-row {
            width: 100%;
            height: 1;
            min-height: 1;
            layout: horizontal;
            background: $background;
            padding: 0 1;
        }

        .transcript-prompt-mark {
            width: 2;
            height: 1;
            color: $text-muted;
        }

        #home-prompt {
            dock: bottom;
            width: 1fr;
            border: tall $primary;
        }

        #prompt, #editor-aux-chat-prompt {
            width: 1fr;
            height: 1;
            min-height: 1;
            border: none;
            background: $background;
            padding: 0;
            color: $text;
        }

        #prompt:focus, #prompt:focus-within,
        #editor-aux-chat-prompt:focus, #editor-aux-chat-prompt:focus-within {
            border: none;
            background: $background;
        }

        #chat-speak, #editor-aux-speak {
            min-width: 8;
            height: 1;
            min-height: 1;
            max-height: 1;
            border: none;
            margin: 0;
            padding: 0 1;
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

        #download-phase, #download-bar, #train-curve, #model-picker-hint, #status-bar {
            height: 1;
            padding: 0 1;
            color: $text-muted;
        }

        #download-phase {
            text-style: italic;
        }

        #status-bar {
            dock: bottom;
            background: $surface;
            display: none;
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
        self.load_progress_bps = 0.0
        self.model_status_text = "loading..."
        self._action_started_at: float | None = None
        self.cwd = sophon_project_root()
        self._stderr_prev = sys.__stderr__
        self._pending_system_lines: list[str] = []
        self._pending_assistant_lines: list = []
        self._pending_speech_clips: list = []
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

    def _call_on_app_thread(self, callback, *args) -> None:
        if getattr(self, "_thread_id", None) == threading.get_ident():
            callback(*args)
            return
        self.call_from_thread(callback, *args)

    def _handle_exception(self, error: Exception) -> None:
        if self.session_log is not None:
            import traceback

            self.session_log.write("err", traceback.format_exc())
        super()._handle_exception(error)

    def _wrap_chat_io(self) -> ChatIo:
        def emit(text: str) -> None:
            if self.session_log is not None:
                self.session_log.write("sys", text)
            self._call_on_app_thread(self.append_system, text)

        def on_assistant(text: str, trace: object | None = None) -> None:
            if self.session_log is not None:
                self.session_log.write("assistant", text)
                if trace is not None:
                    from cli.chat_trace import TurnTrace

                    if isinstance(trace, TurnTrace):
                        for line in format_turn_trace_lines(trace, for_markup=False, reply_text=text):
                            self.session_log.write("trace", line.plain)
            self._call_on_app_thread(self.append_assistant, text, trace)

        def on_dictate(text: str) -> None:
            self._call_on_app_thread(self._dictate_into_prompt, text)

        def on_speech_clip(clip) -> None:
            self._call_on_app_thread(self.append_speech_clip, clip)

        def on_mic(phase: str) -> None:
            self._call_on_app_thread(self._apply_mic_phase, phase)

        def on_clear() -> None:
            self._call_on_app_thread(self.clear_visible_chat)

        def on_step(step) -> None:
            if self.session_log is not None:
                from cli.chat_trace import format_heartbeat_step_plain

                self.session_log.write("step", format_heartbeat_step_plain(step))
            self._call_on_app_thread(self.append_heartbeat_step, step)

        def on_heartbeat_end() -> None:
            self._call_on_app_thread(self.finish_heartbeat)

        def on_permission(request) -> str:
            from harness.approval import DECISION_DENY, PermissionRequest

            if not isinstance(request, PermissionRequest):
                return DECISION_DENY
            future: Future[str] = Future()

            def _show() -> None:
                def _done(choice: str | None) -> None:
                    if not future.done():
                        future.set_result(choice or DECISION_DENY)

                self.push_screen(PermissionPrompt(request), _done)

            self._call_on_app_thread(_show)
            try:
                return future.result()
            except Exception:
                return DECISION_DENY

        def on_setup_choice(summary: str) -> str:
            future: Future[str] = Future()

            def _show() -> None:
                def _done(choice: str | None) -> None:
                    if not future.done():
                        future.set_result(choice or SETUP_SKIP)

                self.push_screen(SetupPrompt(summary), _done)

            self._call_on_app_thread(_show)
            try:
                return future.result()
            except Exception:
                return SETUP_SKIP

        def on_choice(title: str, options: list, timeout_s: float | None) -> str | None:
            future: Future[str | None] = Future()

            def _show() -> None:
                def _done(choice: str | None) -> None:
                    if not future.done():
                        future.set_result(choice)

                self.push_screen(ChoicePrompt(str(title), list(options), timeout_s), _done)

            self._call_on_app_thread(_show)
            try:
                return future.result()
            except Exception:
                return None

        def on_pick_model(backend: str) -> str | None:
            state = self.session_state
            if state is None:
                return None
            entries = [
                entry
                for entry in build_model_picker_entries(state)
                if entry.group == backend
            ]
            if not any(entry.kind != "header" for entry in entries):
                return None
            future: Future[str | None] = Future()

            def _show() -> None:
                def _done(entry: ModelPickerEntry | None) -> None:
                    if future.done():
                        return
                    if entry is None or entry.kind == "header" or not entry.target:
                        future.set_result(None)
                        return
                    target = entry.target
                    name = target.split(":", 1)[1] if ":" in target else target
                    future.set_result(name)

                self.push_screen(ModelPickerScreen(entries), _done)

            self._call_on_app_thread(_show)
            try:
                return future.result()
            except Exception:
                return None

        def on_keybinds_reload() -> None:
            from cli.tui.keybinds import refresh_keybind_surfaces

            def _show() -> None:
                refresh_keybind_surfaces(self)

            self._call_on_app_thread(_show)

        def on_keybinds_edit(path: str) -> None:
            def _show() -> None:
                self.run_worker(self._open_keybinds_file(path), exclusive=True)

            self._call_on_app_thread(_show)

        return ChatIo(
            emit=emit,
            on_assistant=on_assistant,
            on_dictate=on_dictate,
            on_speech_clip=on_speech_clip,
            on_mic=on_mic,
            on_clear=on_clear,
            on_step=on_step,
            on_heartbeat_end=on_heartbeat_end,
            on_permission=on_permission,
            on_setup_choice=on_setup_choice,
            on_choice=on_choice,
            on_pick_model=on_pick_model,
            on_keybinds_reload=on_keybinds_reload,
            on_keybinds_edit=on_keybinds_edit,
        )

    def _apply_mic_phase(self, phase: str) -> None:
        if phase == "stopped":
            return
        screen = self.screen
        handler = getattr(screen, "on_sst_mic_phase", None)
        if callable(handler):
            handler(phase)
            return
        if phase == "max" and self.session_state is not None:
            from cli.chat import finish_sst_recording

            state = self.session_state
            self.run_worker(lambda: finish_sst_recording(state), thread=True, group="sst-mic")

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
        lines = text.split("\n", 1)
        if _ELAPSED_SUFFIX_RE.search(lines[0].rstrip()):
            return text
        elapsed = time.monotonic() - self._action_started_at
        suffix = f" · {_format_action_elapsed(elapsed)}"
        lines[0] = lines[0].rstrip() + suffix
        return "\n".join(lines)

    def copy_to_clipboard(self, text: str) -> None:
        if copy_text_to_system_clipboard(text):
            return
        super().copy_to_clipboard(text)

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
        bytes_per_s: float = 0.0,
    ) -> None:
        self.post_message(
            LoadProgress(status, n=n, total=total, phase=phase, bytes_per_s=bytes_per_s)
        )

    def on_load_progress(self, event: LoadProgress) -> None:
        self.set_load_status(
            event.status,
            n=event.n,
            total=event.total,
            phase=event.phase,
            bytes_per_s=event.bytes_per_s,
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
        bytes_per_s: float = 0.0,
    ) -> None:
        if self.session_state is None or self._action_started_at is not None:
            self.start_action_timer()
        self.load_status = status
        self.load_progress_n = n
        self.load_progress_total = total
        self.load_progress_phase = phase or status
        self.load_progress_bps = bytes_per_s
        self.model_status_text = self.format_with_elapsed(status)
        screen = self.screen
        try:
            if isinstance(screen, DashboardScreen):
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
        self.load_progress_bps = 0.0
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
        from cli.code_assist import ensure_assist

        ensure_assist(state)
        state.assist_ui_hook = self._assist_ui_hook
        self.post_message(SessionReady())
        self.refresh_active_screen()

    def _assist_ui_hook(self) -> None:
        try:
            self.call_from_thread(self._refresh_assist_ui)
        except RuntimeError:
            self._refresh_assist_ui()

    def _refresh_assist_ui(self) -> None:
        screen = self.screen
        if isinstance(screen, EditorScreen):
            screen.refresh_review_pane(focus_tab=True)
            return
        self.notify("Pending code edits. Open Editor and the Review tab.")

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
        if isinstance(screen, DashboardScreen):
            screen.refresh_stats()
        elif isinstance(screen, ChatScreen):
            screen.refresh_from_state()
        elif isinstance(screen, EditorScreen):
            screen.refresh_chat_pane()
            screen.refresh_review_pane()

    async def open_editor(self) -> None:
        self._apply_window_title(EDITOR_TITLE)
        await self.switch_mode("editor")
        self.call_after_refresh(self.refresh_active_screen)

    async def _open_keybinds_file(self, path: str) -> None:
        await self.open_editor()
        screen = self.screen
        opener = getattr(screen, "_request_open_path", None)
        if callable(opener):
            opener(Path(path))

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
        elif nxt == "editor":
            await self.open_editor()
        else:
            await self.open_chat()

    def clear_visible_chat(self) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.clear_transcript()
        elif isinstance(screen, EditorScreen):
            screen.clear_chat_log()
        self._pending_system_lines.clear()
        self._pending_assistant_lines.clear()

    def append_heartbeat_step(self, step) -> None:
        handler = getattr(self.screen, "append_heartbeat_step", None)
        if callable(handler):
            handler(step)

    def finish_heartbeat(self) -> None:
        handler = getattr(self.screen, "finish_heartbeat", None)
        if callable(handler):
            handler()

    def append_system(self, text: str) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.append_system_from_app(text)
        elif isinstance(screen, EditorScreen):
            screen.append_chat_system(text)
        else:
            self._pending_system_lines.append(text)

    def append_assistant(self, text: str, trace: object | None = None) -> None:
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.append_assistant_from_app(text, trace)
        elif isinstance(screen, EditorScreen):
            screen.append_chat_assistant(text, trace)
        else:
            self._pending_assistant_lines.append((text, trace))

    def append_speech_clip(self, clip) -> None:
        if self.session_log is not None:
            self.session_log.write("speech", format_speech_clip_line(clip).plain)
        screen = self.screen
        if isinstance(screen, ChatScreen):
            screen.append_speech_clip(clip)
        elif isinstance(screen, EditorScreen):
            screen.append_speech_clip(clip)
        else:
            self._pending_speech_clips.append(clip)

    def play_session_clip(self, clip_id: int, speed: float = 1.0) -> None:
        state = self.session_state
        if state is None:
            self.notify("No chat session.")
            return
        self.run_worker(
            lambda: play_speech_clip(state, str(int(clip_id)), speed=float(speed)),
            thread=True,
            exclusive=False,
            group="speech-play",
        )

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
        if isinstance(screen, EditorScreen):
            screen.append_terminal_log(text)
        elif isinstance(screen, ChatScreen):
            screen.append_system_from_app(text)
        else:
            self._pending_system_lines.append(text)


def run_tui_app(params: ChatCliParams, session_log: TuiSessionLog | None) -> None:
    from cli.tui.terminal_image import probe_terminal_graphics

    probe_terminal_graphics()
    SophonTuiApp(params, session_log).run()
