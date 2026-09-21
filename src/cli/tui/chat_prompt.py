from __future__ import annotations

from pathlib import Path

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.suggester import Suggester
from textual.widgets import Input, OptionList

from cli.slash_index import slash_query_from_value
from cli.tui.paste_drop import extract_dropped_paths, mention_roots, suggest_at_completion

_NAV_CAPTURED_KEYS = frozenset({"ctrl+d", "ctrl+e", "ctrl+g", "ctrl+h"})


def _input_bindings_without_nav() -> list[Binding]:
    out: list[Binding] = []
    for bind in Input.BINDINGS:
        key = getattr(bind, "key", "") or ""
        parts = [part.strip() for part in str(key).split(",") if part.strip()]
        kept = [part for part in parts if part.lower() not in _NAV_CAPTURED_KEYS]
        if not kept:
            continue
        if kept == parts:
            out.append(bind)
            continue
        out.append(
            Binding(
                ",".join(kept),
                bind.action,
                bind.description,
                show=getattr(bind, "show", False),
                priority=getattr(bind, "priority", False),
            )
        )
    return out


class FilesDropped(Message):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        super().__init__()


class SlashPaletteList(OptionList):
    can_focus = False
    BINDINGS = [
        bind
        for bind in OptionList.BINDINGS
        if "enter" not in (getattr(bind, "key", "") or "").lower()
    ]


class ChatPromptSuggester(Suggester):
    def __init__(self, prompt: "ChatPromptInput") -> None:
        super().__init__(use_cache=False, case_sensitive=True)
        self._prompt = prompt

    async def get_suggestion(self, value: str) -> str | None:
        slash = self._prompt.slash_ghost_value(value)
        if slash:
            return slash
        text = value or ""
        at = text.rfind("@")
        if at < 0:
            return None
        if at > 0 and not text[at - 1].isspace():
            return None
        fragment = text[at + 1 :]
        if " " in fragment or "\t" in fragment:
            return None
        match = suggest_at_completion(fragment, mention_roots())
        if not match:
            return None
        return text[: at + 1] + match


class ChatPromptInput(Input):
    DEFAULT_CSS = """
    ChatPromptInput {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0;
        background: $background;
        color: $text;
    }
    ChatPromptInput:focus {
        border: none;
        background: $background;
    }
    """
    BINDINGS = [
        *_input_bindings_without_nav(),
        Binding("enter", "submit", "Submit", show=False, priority=True),
        Binding("tab", "accept_path_suggestion", "Complete", show=False, priority=True),
        Binding("right", "slash_fill", show=False, priority=True),
        Binding("up", "slash_up", show=False, priority=True),
        Binding("down", "slash_down", show=False, priority=True),
        Binding("escape", "slash_escape", show=False, priority=True),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.suggester = ChatPromptSuggester(self)

    def _slash_host(self):
        node = self.parent
        while node is not None:
            visible = getattr(node, "slash_palette_visible", None)
            if callable(visible):
                return node
            node = getattr(node, "parent", None)
        return None

    def slash_ghost_value(self, value: str | None = None) -> str | None:
        text = self.value if value is None else value
        if slash_query_from_value(text) is None:
            return None
        host = self._slash_host()
        name = None
        if host is not None:
            getter = getattr(host, "slash_highlighted_name", None)
            if callable(getter):
                name = getter()
        if not name:
            return None
        ghost = "/" + str(name)
        if ghost.startswith(text):
            return ghost
        return None

    def apply_slash_ghost(self) -> None:
        ghost = self.slash_ghost_value()
        self._suggestion = ghost or ""
        self.refresh()

    def action_slash_up(self) -> None:
        host = self._slash_host()
        mover = getattr(host, "slash_move", None) if host is not None else None
        if callable(mover) and mover(-1):
            return

    def action_slash_down(self) -> None:
        host = self._slash_host()
        mover = getattr(host, "slash_move", None) if host is not None else None
        if callable(mover) and mover(1):
            return

    def action_slash_escape(self) -> None:
        host = self._slash_host()
        dismiss = getattr(host, "slash_dismiss", None) if host is not None else None
        if callable(dismiss) and dismiss():
            return
        focus = getattr(self.screen, "action_focus_prompt", None)
        if callable(focus):
            focus()

    def action_accept_path_suggestion(self) -> None:
        host = self._slash_host()
        commit = getattr(host, "slash_commit", None) if host is not None else None
        if callable(commit) and commit():
            return
        suggestion = getattr(self, "_suggestion", "") or ""
        if suggestion and getattr(self, "cursor_at_end", False):
            self.value = suggestion
            self.cursor_position = len(self.value)
            return
        cycle = getattr(self.screen, "action_cycle_focus", None)
        if callable(cycle):
            cycle()

    def action_slash_fill(self) -> None:
        host = self._slash_host()
        filler = getattr(host, "slash_fill", None) if host is not None else None
        if callable(filler) and filler():
            return
        self.action_cursor_right()

    async def action_submit(self) -> None:
        host = self._slash_host()
        dismiss = getattr(host, "slash_dismiss", None) if host is not None else None
        if callable(dismiss):
            dismiss()
        await super().action_submit()

    def _on_paste(self, event: events.Paste) -> None:
        paths = extract_dropped_paths(event.text)
        if paths:
            event.stop()
            self.post_message(FilesDropped(paths))
            return
        super()._on_paste(event)
