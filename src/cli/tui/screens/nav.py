from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Static

MODE_CYCLE = ("dashboard", "editor", "chat")

DASHBOARD_TITLE = "sophon"
EDITOR_TITLE = "sophon editor"
CHAT_TITLE = "NEXUS"

NAV_BINDINGS = [
    Binding("ctrl+d", "open_dashboard", "Dashboard", show=False, priority=True),
    Binding("ctrl+e", "open_editor", "Editor", show=False, priority=True),
    Binding("ctrl+g", "open_chat", "Nexus", show=False, priority=True),
    Binding("ctrl+h", "cycle_mode", "Cycle", show=False, priority=True),
]


def compose_nav_bar(active: str) -> ComposeResult:
    with Horizontal(id="nav-bar"):
        yield Button(
            "Dashboard",
            id="nav-dashboard",
            variant="primary" if active == "dashboard" else "default",
        )
        yield Button(
            "Editor",
            id="nav-editor",
            variant="primary" if active == "editor" else "default",
        )
        yield Button(
            "Nexus",
            id="nav-chat",
            variant="primary" if active == "chat" else "default",
        )
        yield Static("", id="nav-spacer")


class ModeNavigationMixin:
    async def action_open_dashboard(self) -> None:
        await self.app.open_dashboard()

    async def action_open_editor(self) -> None:
        await self.app.open_editor()

    async def action_open_chat(self) -> None:
        await self.app.open_chat()

    async def action_cycle_mode(self) -> None:
        await self.app.cycle_mode()

    def handle_nav_button(self, button_id: str | None) -> bool:
        if button_id == "nav-dashboard":
            self.run_worker(self.app.open_dashboard, exclusive=True)
            return True
        if button_id == "nav-editor":
            self.run_worker(self.app.open_editor, exclusive=True)
            return True
        if button_id == "nav-chat":
            self.run_worker(self.app.open_chat, exclusive=True)
            return True
        return False
