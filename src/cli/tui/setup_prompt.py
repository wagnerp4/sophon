from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

SETUP_APPLY = "apply"
SETUP_SKIP = "skip"


class SetupPrompt(ModalScreen[str]):
    BINDINGS = [
        Binding("1", "choose_apply", "Install", show=True),
        Binding("2", "choose_skip", "Skip", show=True),
        Binding("escape", "choose_skip", "Skip", show=False),
    ]

    DEFAULT_CSS = """
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
    """

    def __init__(self, summary: str) -> None:
        super().__init__()
        self._summary = summary

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-dialog"):
            yield Static("/setup", id="setup-title")
            yield Static(self._summary or "install rank 1, or skip", id="setup-body")
            with Horizontal(id="setup-actions"):
                yield Button("1 install", id="setup-apply", variant="primary")
                yield Button("2 skip", id="setup-skip", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "setup-apply": SETUP_APPLY,
            "setup-skip": SETUP_SKIP,
        }
        choice = mapping.get(event.button.id or "")
        if choice:
            self.dismiss(choice)

    def action_choose_apply(self) -> None:
        self.dismiss(SETUP_APPLY)

    def action_choose_skip(self) -> None:
        self.dismiss(SETUP_SKIP)
