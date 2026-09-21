from __future__ import annotations

import time

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical, VerticalGroup, VerticalScroll
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from cli.chat_display import (
    TranscriptLine,
    format_assistant_turn_lines,
    format_speech_clip_line,
    format_system_line,
)
from cli.slash_index import (
    EMPTY_QUERY_LIMIT,
    format_slash_row,
    rank_slash_commands,
    slash_query_from_value,
)
from cli.tui.chat_prompt import ChatPromptInput, SlashPaletteList
from cli.chat_trace import (
    HeartbeatStep,
    format_heartbeat_step_plain,
    heartbeat_kind_count_label,
    heartbeat_thinking_title,
)

from processing.audio.speech.session_store import (
    SpeechClip,
    default_replay_speeds,
    format_replay_speed,
)

HOLD_TO_TALK_S = 0.28


def speed_token(speed: float) -> str:
    return format_replay_speed(speed).replace(".", "p").replace("x", "")


def parse_speed_token(token: str) -> float:
    return float(token.replace("p", "."))


class SpeechClipBar(Horizontal):
    def __init__(self, clip: SpeechClip) -> None:
        super().__init__(classes="speech-clip-bar")
        self.clip = clip

    def compose(self) -> ComposeResult:
        who = "you" if self.clip.role == "user" else "nexus"
        yield Static(
            f"[dim]clip {self.clip.clip_id} {who} {self.clip.duration_s:.1f}s[/dim]",
            classes="speech-clip-label",
        )
        for speed in default_replay_speeds():
            yield Button(
                format_replay_speed(speed),
                id=f"speech-play-{self.clip.clip_id}-{speed_token(speed)}",
                classes="speech-clip-btn",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        prefix = f"speech-play-{self.clip.clip_id}-"
        if not button_id.startswith(prefix):
            return
        speed = parse_speed_token(button_id[len(prefix) :])
        play = getattr(self.app, "play_session_clip", None)
        if callable(play):
            play(self.clip.clip_id, speed)


class ChatLogPane(Vertical):
    def __init__(
        self,
        *,
        transcript_id: str = "transcript",
        prompt_id: str = "prompt",
        **kwargs,
    ) -> None:
        self._transcript_id = transcript_id
        self._prompt_id = prompt_id
        super().__init__(**kwargs)
        self.add_class("chat-log-wrap")
        self._slash_rows: list[str | None] = []
        self._slash_open = False

    def compose(self) -> ComposeResult:
        yield ChatTranscript(id=self._transcript_id, prompt_id=self._prompt_id)
        yield SlashPaletteList(id=self._prompt_id + "-slash-palette", classes="slash-palette")
        with Horizontal(classes="transcript-prompt-row"):
            yield Static(">", classes="transcript-prompt-mark")
            yield ChatPromptInput(
                placeholder="Message, /command, @path, or ! for shell",
                id=self._prompt_id,
                disabled=True,
            )

    def on_mount(self) -> None:
        listing = self._slash_palette()
        if listing is not None:
            listing.display = False

    def _slash_palette(self) -> SlashPaletteList | None:
        try:
            return self.query_one("#" + self._prompt_id + "-slash-palette", SlashPaletteList)
        except NoMatches:
            return None

    def _slash_prompt(self) -> ChatPromptInput | None:
        try:
            return self.query_one("#" + self._prompt_id, ChatPromptInput)
        except NoMatches:
            return None

    def _slash_recents(self) -> list[str]:
        state = getattr(self.app, "session_state", None)
        recents = getattr(state, "slash_recents", None)
        if isinstance(recents, list):
            return list(recents)
        return []

    def slash_palette_visible(self) -> bool:
        return self._slash_open and bool(self._slash_rows)

    def slash_highlighted_name(self) -> str | None:
        listing = self._slash_palette()
        if listing is None or not self._slash_rows:
            return None
        index = listing.highlighted
        if index is None or index < 0 or index >= len(self._slash_rows):
            for name in self._slash_rows:
                if name:
                    return name
            return None
        name = self._slash_rows[index]
        if name:
            return name
        for name in self._slash_rows:
            if name:
                return name
        return None

    def slash_dismiss(self) -> bool:
        if not self.slash_palette_visible():
            return False
        self._hide_slash_palette()
        prompt = self._slash_prompt()
        if prompt is not None:
            prompt.apply_slash_ghost()
        return True

    def slash_move(self, delta: int) -> bool:
        if not self.slash_palette_visible():
            return False
        listing = self._slash_palette()
        if listing is None:
            return False
        count = len(self._slash_rows)
        if count <= 0:
            return False
        current = listing.highlighted
        if current is None:
            current = 0
        idx = current
        for _ in range(count):
            idx = (idx + delta) % count
            if self._slash_rows[idx]:
                listing.highlighted = idx
                prompt = self._slash_prompt()
                if prompt is not None:
                    prompt.apply_slash_ghost()
                return True
        return False

    def slash_fill(self) -> bool:
        if not self.slash_palette_visible():
            return False
        prompt = self._slash_prompt()
        if prompt is None or not getattr(prompt, "cursor_at_end", False):
            return False
        name = self.slash_highlighted_name()
        if not name:
            return False
        filled = "/" + name
        if prompt.value == filled:
            return False
        prompt.value = filled
        prompt.cursor_position = len(filled)
        return True

    def slash_commit(self) -> bool:
        if not self.slash_palette_visible():
            return False
        name = self.slash_highlighted_name()
        prompt = self._slash_prompt()
        if not name or prompt is None:
            return False
        prompt.value = "/" + name + " "
        prompt.cursor_position = len(prompt.value)
        self._hide_slash_palette()
        prompt.apply_slash_ghost()
        return True

    def _hide_slash_palette(self) -> None:
        listing = self._slash_palette()
        self._slash_rows = []
        self._slash_open = False
        if listing is None:
            return
        listing.clear_options()
        listing.display = False

    def _sync_slash_palette(self, value: str) -> None:
        listing = self._slash_palette()
        if listing is None:
            return
        query = slash_query_from_value(value)
        if query is None:
            self._hide_slash_palette()
            return
        hits = rank_slash_commands(query, self._slash_recents(), limit=EMPTY_QUERY_LIMIT)
        if not hits:
            self._hide_slash_palette()
            return
        sections = {entry.section for entry in hits}
        options: list[Option] = []
        rows: list[str | None] = []
        last_section = ""
        grouped = len(sections) > 1
        for entry in hits:
            if grouped and entry.section != last_section:
                last_section = entry.section
                options.append(
                    Option(
                        "  " + entry.section,
                        id="slash-h-" + entry.section + "-" + str(len(rows)),
                        disabled=True,
                    )
                )
                rows.append(None)
            options.append(
                Option(
                    format_slash_row(entry),
                    id="slash-cmd-" + entry.name,
                )
            )
            rows.append(entry.name)
        listing.clear_options()
        listing.add_options(options)
        self._slash_rows = rows
        listing.display = True
        self._slash_open = True
        listing.styles.height = min(len(options) + 1, 12)
        first = 0
        while first < len(rows) and not rows[first]:
            first += 1
        if first < len(rows):
            listing.highlighted = first
        prompt = self._slash_prompt()
        if prompt is not None:
            prompt.apply_slash_ghost()

    def on_input_changed(self, event: Input.Changed) -> None:
        widget = getattr(event, "input", None) or getattr(event, "control", None)
        if getattr(widget, "id", None) != self._prompt_id:
            return
        value = getattr(event, "value", None)
        if value is None:
            value = str(getattr(widget, "value", "") or "")
        self._sync_slash_palette(value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        listing = self._slash_palette()
        source = getattr(event, "option_list", None) or getattr(event, "control", None)
        if listing is None or source is not listing:
            return
        event.stop()
        self.slash_commit()


class ChatTranscript(VerticalScroll):
    can_focus = True
    BINDINGS = [
        Binding("enter", "focus_inline_prompt", "Input", show=False, priority=True),
    ]

    def __init__(self, prompt_id: str = "prompt", **kwargs) -> None:
        self._prompt_id = prompt_id
        super().__init__(**kwargs)

    def prompt(self) -> ChatPromptInput | None:
        try:
            return self.screen.query_one("#" + self._prompt_id, ChatPromptInput)
        except NoMatches:
            return None

    def _mount_entry(self, widget) -> None:
        self.mount(widget)
        self.scroll_end(animate=False)

    def write_markup(self, markup: str) -> None:
        self._mount_entry(Static(markup if markup else " ", markup=True, classes="transcript-line"))

    def write_clip(self, clip: SpeechClip) -> None:
        self._mount_entry(SpeechClipBar(clip))

    def write_heartbeat(self) -> "TurnHeartbeat":
        widget = TurnHeartbeat()
        self._mount_entry(widget)
        return widget

    def clear_lines(self) -> None:
        for child in list(self.children):
            child.remove()

    def _focus_prompt(self) -> None:
        prompt = self.prompt()
        if prompt is not None and not prompt.disabled:
            prompt.focus()

    def action_focus_inline_prompt(self) -> None:
        self._focus_prompt()

    def on_click(self, event: events.Click) -> None:
        target = event.widget
        if target is None:
            return
        if isinstance(target, (Button, ChatPromptInput)):
            return
        classes = getattr(target, "classes", ())
        if "heartbeat-title" in classes or "heartbeat-kind-title" in classes:
            return
        self._focus_prompt()

    def on_key(self, event: events.Key) -> None:
        if not event.is_printable or event.character is None:
            return
        prompt = self.prompt()
        if prompt is None or prompt.disabled:
            return
        prompt.focus()
        prompt.insert_text_at_cursor(event.character)
        event.prevent_default()
        event.stop()


class HeartbeatKindGroup(VerticalGroup):
    def __init__(self, kind: str) -> None:
        super().__init__(classes="heartbeat-kind")
        self._kind = kind
        self._count = 0
        self._lines: list[str] = []
        self._expanded = False
        self._heading = Static(" ", markup=False, classes="heartbeat-kind-title")
        self._body = Static(" ", markup=True, classes="heartbeat-body")
        self._body.display = False
        self._refresh_heading()

    def compose(self) -> ComposeResult:
        yield self._heading
        yield self._body

    def on_click(self, event: events.Click) -> None:
        if event.widget is not self._heading:
            return
        event.stop()
        self._expanded = not self._expanded
        self._body.display = self._expanded
        self._refresh_heading()

    def _refresh_heading(self) -> None:
        mark = "▼" if self._expanded else "▶"
        n = self._count if self._count > 0 else 1
        self._heading.update(mark + " " + heartbeat_kind_count_label(self._kind, n))

    def add_step(self, step: HeartbeatStep) -> None:
        from rich.markup import escape

        self._count += 1
        self._lines.append(escape(format_heartbeat_step_plain(step)))
        self._body.update("[dim]" + "\n".join(self._lines) + "[/]")
        self._refresh_heading()


class TurnHeartbeat(VerticalGroup):
    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    _REASONING_MAX = 2400

    def __init__(self) -> None:
        super().__init__(classes="turn-heartbeat")
        self._steps: list[HeartbeatStep] = []
        self._running = True
        self._started = time.monotonic()
        self._applied = 0
        self._think_parts: list[str] = []
        self._groups: dict[str, HeartbeatKindGroup] = {}
        self._expanded = True
        self._heading = Static("▼ Thinking", markup=False, classes="heartbeat-title")
        self._reasoning = Static("", markup=True, classes="heartbeat-reasoning")
        self._reasoning.display = False
        self._stack = VerticalGroup(classes="heartbeat-stack")

    def compose(self) -> ComposeResult:
        yield self._heading
        yield self._reasoning
        yield self._stack

    def on_mount(self) -> None:
        self._flush_steps()
        self.pulse(0)

    def on_click(self, event: events.Click) -> None:
        if event.widget is not self._heading:
            return
        event.stop()
        self._expanded = not self._expanded
        self._reasoning.display = self._expanded and bool(self._think_parts)
        self._stack.display = self._expanded
        self.pulse(0)

    def add_step(self, step: HeartbeatStep) -> None:
        self._steps.append(step)
        self._flush_steps()
        self.pulse(0)
        parent = self.parent
        scroll = getattr(parent, "scroll_end", None)
        if callable(scroll):
            scroll(animate=False)

    def pulse(self, spinner_i: int) -> None:
        elapsed = time.monotonic() - self._started
        if self._running:
            frame = self._FRAMES[spinner_i % len(self._FRAMES)]
            label = heartbeat_thinking_title(
                self._steps, running=True, elapsed_s=elapsed, frame=frame
            )
        else:
            label = heartbeat_thinking_title(
                self._steps, running=False, elapsed_s=elapsed, frame=""
            )
        mark = "▼" if self._expanded else "▶"
        self._heading.update(mark + " " + label)

    def finish(self) -> None:
        self._running = False
        self.pulse(0)

    def _flush_steps(self) -> None:
        if not self.is_mounted:
            return
        while self._applied < len(self._steps):
            self._apply_step(self._steps[self._applied])
            self._applied += 1

    def _apply_step(self, step: HeartbeatStep) -> None:
        kind = (step.kind or "").strip().lower()
        if kind == "think":
            self._apply_think(step)
            return
        if not kind:
            kind = "unknown"
        group = self._groups.get(kind)
        if group is None:
            group = HeartbeatKindGroup(kind)
            self._groups[kind] = group
            group.add_step(step)
            self._stack.mount(group)
            return
        group.add_step(step)

    def _apply_think(self, step: HeartbeatStep) -> None:
        from rich.markup import escape

        text = (step.preview or "").strip()
        if not text:
            return
        if text in self._think_parts:
            return
        self._think_parts.append(text)
        shown = "\n\n".join(self._think_parts)
        if len(shown) > self._REASONING_MAX:
            shown = "…" + shown[-self._REASONING_MAX :]
        self._reasoning.update("[dim]" + escape(shown) + "[/]")
        if self._expanded:
            self._reasoning.display = True


class TranscriptHostMixin:
    _transcript_id = "transcript"

    def _init_transcript_host(self) -> None:
        self._transcript_plain: list[str] = []
        self._turn_counter = 0
        self._heartbeat: TurnHeartbeat | None = None
        self._heartbeat_active = False
        self._spinner_i = 0

    def _transcript_widget(self) -> ChatTranscript | None:
        try:
            return self.query_one("#" + self._transcript_id, ChatTranscript)
        except NoMatches:
            return None

    def _write_transcript(self, line: TranscriptLine) -> None:
        self._transcript_plain.append(line.plain)
        log = self._transcript_widget()
        if log is None:
            return
        if line.plain:
            log.write_markup(line.markup)
        else:
            log.write_markup("")

    def _append_system(self, text: str) -> None:
        for chunk in text.splitlines() or [""]:
            self._write_transcript(format_system_line(chunk, for_markup=True))

    def _append_assistant(self, text: str, trace: object | None = None) -> None:
        from cli.chat_trace import TurnTrace

        turn_trace = trace if isinstance(trace, TurnTrace) else None
        hide_trace = self._heartbeat_active
        for line in format_assistant_turn_lines(
            text,
            turn=self._turn_counter,
            trace=turn_trace,
            for_markup=True,
            include_trace_details=not hide_trace,
        ):
            self._write_transcript(line)
        self._heartbeat = None
        self._heartbeat_active = False

    def clear_transcript(self) -> None:
        self._turn_counter = 0
        self._transcript_plain.clear()
        self._heartbeat = None
        self._heartbeat_active = False
        log = self._transcript_widget()
        if log is not None:
            log.clear_lines()

    def _ensure_heartbeat(self) -> TurnHeartbeat | None:
        if self._heartbeat is not None and self._heartbeat._running:
            return self._heartbeat
        log = self._transcript_widget()
        if log is None:
            return None
        self._heartbeat = log.write_heartbeat()
        self._heartbeat_active = True
        return self._heartbeat

    def append_heartbeat_step(self, step) -> None:
        widget = self._ensure_heartbeat()
        if widget is None:
            return
        if isinstance(step, HeartbeatStep):
            widget.add_step(step)
            self._transcript_plain.append(format_heartbeat_step_plain(step))

    def finish_heartbeat(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.finish()

    def _pulse_heartbeat(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.pulse(self._spinner_i)

    def append_speech_clip(self, clip) -> None:
        line = format_speech_clip_line(clip, for_markup=False)
        self._transcript_plain.append(line.plain)
        log = self._transcript_widget()
        if log is not None:
            log.write_clip(clip)

    def action_copy_transcript(self) -> None:
        from cli.tui.io import copy_text_to_system_clipboard

        payload = "\n".join(self._transcript_plain)
        notify = getattr(self, "notify", None)
        if not payload.strip():
            if callable(notify):
                notify("Nothing to copy.")
            return
        if copy_text_to_system_clipboard(payload):
            if callable(notify):
                notify("Copied chat to clipboard.")
            return
        app = getattr(self, "app", None)
        copy = getattr(app, "copy_to_clipboard", None)
        if callable(copy):
            copy(payload)
        if callable(notify):
            notify("Could not copy chat to the system clipboard.")

    def action_copy_chat(self) -> None:
        self.action_copy_transcript()


class SpeakButton(Button):
    DEFAULT_CSS = """
    SpeakButton {
        min-width: 8;
        width: auto;
        height: 1;
        min-height: 1;
        max-height: 1;
        border: none;
        padding: 0 1;
    }
    """

    def __init__(self, button_id: str) -> None:
        super().__init__("Speak", id=button_id, variant="default", classes="speak-btn")
        self._down_at: float | None = None

    def set_phase(self, phase: str) -> None:
        if phase == "recording":
            self.label = "Stop"
            self.variant = "error"
            self.disabled = False
            return
        if phase == "transcribing":
            self.label = "Wait"
            self.variant = "warning"
            self.disabled = True
            return
        self.label = "Speak"
        self.variant = "default"
        self.disabled = False

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        event.stop()
        self._down_at = time.monotonic()
        handler = getattr(self.screen, "begin_speak_hold", None)
        if callable(handler):
            handler()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if event.button != 1:
            return
        event.stop()
        down = self._down_at
        self._down_at = None
        held = (time.monotonic() - down) if down is not None else 0.0
        handler = getattr(self.screen, "end_speak_hold", None)
        if callable(handler):
            handler(held)

    def on_click(self, event: events.Click) -> None:
        event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()


class PushToTalkMixin:
    _mic_hold_started: bool = False
    _mic_transcribing: bool = False

    def sst_input_blocked(self) -> bool:
        return False

    def sst_recording(self) -> bool:
        from cli.chat import sst_recording

        state = getattr(self.app, "session_state", None)
        return bool(state is not None and sst_recording(state))

    def _speak_button(self) -> SpeakButton | None:
        try:
            return self.query_one(".speak-btn", SpeakButton)
        except NoMatches:
            return None

    def set_speak_phase(self, phase: str) -> None:
        button = self._speak_button()
        if button is not None:
            button.set_phase(phase)

    def on_sst_mic_phase(self, phase: str) -> None:
        if phase == "recording":
            self.set_speak_phase("recording")
            return
        if phase == "transcribing":
            self._mic_transcribing = True
            self.set_speak_phase("transcribing")
            return
        if phase == "max":
            self._stop_sst_mic()
            return
        self._mic_transcribing = False
        self.set_speak_phase("idle")

    def action_listen(self) -> None:
        if self._mic_transcribing:
            return
        if self.sst_recording():
            self._stop_sst_mic()
            return
        if self.sst_input_blocked():
            return
        self._start_sst_mic()

    def begin_speak_hold(self) -> None:
        if self._mic_transcribing:
            return
        if self.sst_recording():
            self._mic_hold_started = False
            return
        if self.sst_input_blocked():
            return
        self._mic_hold_started = True
        self._start_sst_mic()

    def end_speak_hold(self, held: float) -> None:
        if self._mic_transcribing:
            return
        if self._mic_hold_started:
            self._mic_hold_started = False
            if held >= HOLD_TO_TALK_S and self.sst_recording():
                self._stop_sst_mic()
            return
        if self.sst_recording():
            self._stop_sst_mic()

    def _start_sst_mic(self) -> None:
        from cli.chat import start_sst_recording

        state = getattr(self.app, "session_state", None)
        if state is None:
            return
        start_sst_recording(state)
        self.set_speak_phase("recording")

    def _stop_sst_mic(self) -> None:
        from cli.chat import finish_sst_recording

        state = getattr(self.app, "session_state", None)
        if state is None:
            return
        if self._mic_transcribing:
            return
        self._mic_transcribing = True
        self.set_speak_phase("transcribing")
        busy = getattr(self, "_set_busy", None)
        if callable(busy):
            busy(True, "Transcribing…")
        else:
            chat_busy = getattr(self, "_set_chat_busy", None)
            if callable(chat_busy):
                chat_busy(True)
        app = self.app

        def work() -> None:
            try:
                finish_sst_recording(state)
            finally:
                done = getattr(self, "_sst_finish_done", None)
                if callable(done):
                    app.call_from_thread(done)

        self.run_worker(work, thread=True, exclusive=True, group="sst-mic")

    def _sst_finish_done(self) -> None:
        state = getattr(self.app, "session_state", None)
        if state is not None and bool(getattr(state, "sst_busy", False)):
            return
        self._mic_transcribing = False
        busy = getattr(self, "_set_busy", None)
        if callable(busy):
            busy(False)
        else:
            chat_busy = getattr(self, "_set_chat_busy", None)
            if callable(chat_busy):
                chat_busy(False)
        if not self.sst_recording():
            self.set_speak_phase("idle")

    def _cancel_sst_mic(self) -> None:
        from cli.chat import cancel_sst_recording

        state = getattr(self.app, "session_state", None)
        if state is None:
            return
        cancel_sst_recording(state)
        self._mic_hold_started = False
        self._mic_transcribing = False
        self.set_speak_phase("idle")


# TODO(sst): show a live RMS meter on SpeakButton while recording
