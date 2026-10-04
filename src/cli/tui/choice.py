from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static


class ChoicePrompt(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    DEFAULT_CSS = """
    ChoicePrompt {
        align: center middle;
        background: $background 60%;
    }

    #choice-dialog {
        width: 72;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #choice-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #choice-list {
        height: auto;
        max-height: 16;
    }

    #choice-hint {
        margin-top: 1;
    }
    """

    def __init__(
        self,
        title: str,
        options: list[tuple[str, str, bool]],
        timeout_s: float | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._options = options
        self._timeout_s = timeout_s

    def compose(self) -> ComposeResult:
        with Vertical(id="choice-dialog"):
            yield Static(self._title, id="choice-title")
            yield ListView(id="choice-list")
            hint = "Enter selects. Esc cancels."
            if self._timeout_s and self._timeout_s > 0:
                hint = f"{hint} Stops after {int(self._timeout_s)}s."
            yield Static(hint, id="choice-hint")

    def on_mount(self) -> None:
        panel = self.query_one("#choice-list", ListView)
        for key, label, enabled in self._options:
            item = ListItem(Label(label))
            item.disabled = not enabled
            setattr(item, "choice_key", key if enabled else "")
            panel.append(item)
        panel.focus()
        if self._timeout_s and self._timeout_s > 0:
            self.set_timer(self._timeout_s, self.action_cancel)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        key = str(getattr(event.item, "choice_key", "") or "")
        if not key:
            return
        self.dismiss(key)

    def action_cancel(self) -> None:
        self.dismiss(None)
