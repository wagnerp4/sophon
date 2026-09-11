from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Static

MODE_CYCLE = ("dashboard", "workshop", "editor", "chat")

DASHBOARD_TITLE = "orodruin"
WORKSHOP_TITLE = "orodruin workshop"
EDITOR_TITLE = "orodruin editor"
CHAT_TITLE = "orodruin chat"

NAV_BINDINGS = [
    Binding("ctrl+d", "open_dashboard", "Dashboard", show=True),
    Binding("ctrl+w", "open_workshop", "Workshop", show=True),
    Binding("ctrl+e", "open_editor", "Editor", show=True),
    Binding("ctrl+g", "open_chat", "Chat", show=True),
    Binding("ctrl+h", "cycle_mode", "Cycle", show=True),
]


def compose_nav_bar(active: str) -> ComposeResult:
    with Horizontal(id="nav-bar"):
        yield Button(
            "Dashboard",
            id="nav-dashboard",
            variant="primary" if active == "dashboard" else "default",
        )
        yield Button(
            "Workshop",
            id="nav-workshop",
            variant="primary" if active == "workshop" else "default",
        )
        yield Button(
            "Editor",
            id="nav-editor",
            variant="primary" if active == "editor" else "default",
        )
        yield Button(
            "Chat",
            id="nav-chat",
            variant="primary" if active == "chat" else "default",
        )
        yield Static("", id="nav-spacer")


class ModeNavigationMixin:
    async def action_open_dashboard(self) -> None:
        await self.app.open_dashboard()

    async def action_open_workshop(self) -> None:
        await self.app.open_workshop()

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
        if button_id == "nav-workshop":
            self.run_worker(self.app.open_workshop, exclusive=True)
            return True
        if button_id == "nav-editor":
            self.run_worker(self.app.open_editor, exclusive=True)
            return True
        if button_id == "nav-chat":
            self.run_worker(self.app.open_chat, exclusive=True)
            return True
        return False
