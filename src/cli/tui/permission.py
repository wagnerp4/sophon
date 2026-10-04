from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from harness.approval import DECISION_DENY, DECISION_ONCE, DECISION_PERSIST, PermissionRequest
from harness.prompt import format_permission_prompt

_BUTTONS = (
    ("permission-once", "permission_once", "this time", "primary", DECISION_ONCE, "choose_once"),
    ("permission-persist", "permission_persist", "allow-list", "success", DECISION_PERSIST, "choose_persist"),
    ("permission-deny", "permission_deny", "decline", "warning", DECISION_DENY, "choose_deny"),
)


class PermissionPrompt(ModalScreen[str]):
    BINDINGS = [
        Binding("1", "choose_once", "This time", show=False, priority=True),
        Binding("2", "choose_persist", "Allow-list", show=False, priority=True),
        Binding("3", "choose_deny", "Decline", show=False, priority=True),
        Binding("left", "focus_prev", "Previous", show=False, priority=True),
        Binding("right", "focus_next", "Next", show=False, priority=True),
        Binding("enter", "activate_focused", "Choose", show=False, priority=True),
        Binding("escape", "choose_deny", "Decline", show=False, priority=True),
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

    #permission-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }

    #permission-buttons Button {
        margin: 0 1;
        min-width: 16;
    }

    #permission-hints {
        height: 1;
        align: center middle;
    }

    .permission-hint {
        width: 16;
        height: 1;
        content-align: center middle;
        margin: 0 1;
    }
    """

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self._request = request
        self._order = [item[0] for item in _BUTTONS]
        self._chosen = False
        self._key_choices: dict[str, str] = {
            "1": DECISION_ONCE,
            "2": DECISION_PERSIST,
            "3": DECISION_DENY,
        }

    def _choose(self, choice: str) -> None:
        if self._chosen:
            return
        self._chosen = True
        self.dismiss(choice)

    def on_key(self, event: Key) -> None:
        choice = self._key_choices.get(str(event.key))
        if not choice:
            return
        event.stop()
        event.prevent_default()
        self._choose(choice)

    def _keys(self) -> dict[str, str]:
        from cli.tui.keybinds import format_key_compact, permission_keys

        raw = permission_keys()
        return {name: format_key_compact(key) or key for name, key in raw.items()}

    def compose(self) -> ComposeResult:
        keys = self._keys()
        with Vertical(id="permission-dialog"):
            yield Static("Harness permission", id="permission-title")
            yield Static(
                format_permission_prompt(
                    self._request,
                    once=keys.get("permission_once") or "1",
                    persist=keys.get("permission_persist") or "2",
                    deny=keys.get("permission_deny") or "3",
                ),
                id="permission-body",
            )
            with Horizontal(id="permission-buttons"):
                for button_id, _action, label, variant, _decision, _method in _BUTTONS:
                    yield Button(label, id=button_id, variant=variant)
            with Horizontal(id="permission-hints"):
                for _button_id, action, _label, _variant, _decision, _method in _BUTTONS:
                    yield Static(keys.get(action) or "-", classes="permission-hint")

    def on_mount(self) -> None:
        from cli.tui.keybinds import permission_keys

        methods = {item[1]: item[5] for item in _BUTTONS}
        for action, key in permission_keys().items():
            method = methods.get(action)
            if not method or not key:
                continue
            try:
                self.bind(key, method, show=False, priority=True)
                self._key_choices[str(key)] = {
                    "choose_once": DECISION_ONCE,
                    "choose_persist": DECISION_PERSIST,
                    "choose_deny": DECISION_DENY,
                }[method]
            except Exception:
                continue
        button = self.query_one("#permission-once", Button)
        button.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {item[0]: item[4] for item in _BUTTONS}
        choice = mapping.get(event.button.id or "")
        if choice:
            self._choose(choice)

    def _focused_index(self) -> int:
        focused = self.focused
        for index, button_id in enumerate(self._order):
            if getattr(focused, "id", None) == button_id:
                return index
        return 0

    def _focus_index(self, index: int) -> None:
        button_id = self._order[index % len(self._order)]
        self.query_one("#" + button_id, Button).focus()

    def action_focus_prev(self) -> None:
        self._focus_index(self._focused_index() - 1)

    def action_focus_next(self) -> None:
        self._focus_index(self._focused_index() + 1)

    def action_activate_focused(self) -> None:
        focused = self.focused
        mapping = {item[0]: item[4] for item in _BUTTONS}
        choice = mapping.get(getattr(focused, "id", None) or "")
        if choice:
            self._choose(choice)

    def action_choose_once(self) -> None:
        self._choose(DECISION_ONCE)

    def action_choose_persist(self) -> None:
        self._choose(DECISION_PERSIST)

    def action_choose_deny(self) -> None:
        self._choose(DECISION_DENY)
