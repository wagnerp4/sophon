from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from harness.approval import DECISION_DENY, DECISION_ONCE, DECISION_PERSIST, PermissionRequest
from harness.prompt import format_permission_prompt


class PermissionPrompt(ModalScreen[str]):
    BINDINGS = [
        Binding("1", "choose_once", "This time", show=True),
        Binding("2", "choose_persist", "Allow-list", show=True),
        Binding("3", "choose_deny", "Decline", show=True),
        Binding("escape", "choose_deny", "Decline", show=False),
    ]

    DEFAULT_CSS = """
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
        max-height: 24;
        overflow-y: auto;
    }

    #permission-actions {
        height: auto;
        margin-top: 1;
        align: center middle;
    }

    #permission-actions Button {
        margin: 0 1;
    }
    """
    BINDINGS = [
        Binding("1", "choose_once", "This time", show=True),
        Binding("2", "choose_persist", "Allow-list", show=True),
        Binding("3", "choose_deny", "Decline", show=True),
        Binding("escape", "choose_deny", "Decline", show=False),
    ]

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self._request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-dialog"):
            yield Static("Harness permission", id="permission-title")
            yield Static(format_permission_prompt(self._request), id="permission-body")
            with Horizontal(id="permission-actions"):
                yield Button("1 this time", id="permission-once", variant="primary")
                yield Button("2 allow-list", id="permission-persist", variant="success")
                yield Button("3 decline", id="permission-deny", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "permission-once": DECISION_ONCE,
            "permission-persist": DECISION_PERSIST,
            "permission-deny": DECISION_DENY,
        }
        choice = mapping.get(event.button.id or "")
        if choice:
            self.dismiss(choice)

    def action_choose_once(self) -> None:
        self.dismiss(DECISION_ONCE)

    def action_choose_persist(self) -> None:
        self.dismiss(DECISION_PERSIST)

    def action_choose_deny(self) -> None:
        self.dismiss(DECISION_DENY)
