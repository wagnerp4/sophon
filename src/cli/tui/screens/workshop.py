from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, DirectoryTree, Footer, Header, Input, MarkdownViewer, Static, TextArea
from textual.widgets.directory_tree import DirEntry

from cli.tui.screens.nav import NAV_BINDINGS, ModeNavigationMixin, compose_nav_bar
from cli.tui.tiles.host import format_host_snapshot

_MARKUP_RE = re.compile(r"\[/?[^\]]+\]")
_MARKDOWN_SUFFIXES = {".md", ".markdown"}


def _strip_markup(text: str) -> str:
    return _MARKUP_RE.sub("", text)


class ProjectFileActivated(Message):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__()


class ProjectTree(DirectoryTree):
    auto_expand = False

    def _populate_node(self, node, content: Iterable[Path]) -> None:
        first_load = len(node.children) == 0
        was_expanded = node.is_expanded
        stay_open = was_expanded or first_load
        node.remove_children()
        for path in content:
            node.add(
                path.name,
                data=DirEntry(path),
                allow_expand=self._safe_is_dir(path),
            )
        if stay_open and not was_expanded:
            node.expand()

    async def _on_tree_node_selected(self, event) -> None:
        node = event.node
        if node is None or not node.allow_expand:
            return
        if node.data is not None:
            node.data.loaded = False
        node.expand()

    async def _on_click(self, event: events.Click) -> None:
        if event.chain != 2:
            return
        meta = event.style.meta
        line_no = meta.get("line") if meta else None
        node = self.get_node_at_line(line_no) if isinstance(line_no, int) else None
        if node is None or node.data is None:
            return
        path = node.data.path
        if not path.is_file():
            return
        self.post_message(ProjectFileActivated(path.resolve()))
        event.stop()


class WorkshopScreen(ModeNavigationMixin, Screen):
    BINDINGS = [
        *NAV_BINDINGS,
        Binding("ctrl+c", "absorb_ctrl_c", show=False, priority=True),
        Binding("escape", "leave_content_view", "Input", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.shell_mode = False
        self._log_lines: list[str] = []
        self._view_mode = "log"
        self._markdown_viewer: MarkdownViewer | None = None

    def compose(self) -> ComposeResult:
        app = self.app
        yield Header(show_clock=True)
        yield from compose_nav_bar("workshop")
        with Horizontal(id="home-row"):
            with Vertical(id="home-left"):
                yield Static("Project", id="home-project-title")
                yield ProjectTree(str(app.cwd), id="home-tree")
            with Vertical(id="home-main"):
                yield Static("Loading system stats...", id="home-stats")
                with Vertical(id="home-content"):
                    with Horizontal(id="home-content-toolbar"):
                        yield Static("Shell log", id="home-content-title")
                        yield Button("Copy", id="home-copy", variant="default")
                        yield Button("Shell log", id="home-back", variant="default")
                    with Vertical(id="home-content-body"):
                        yield TextArea("", id="home-log", read_only=True, show_line_numbers=False)
                yield Input(
                    placeholder="Type !command, $command, /shell, or chat",
                    id="home-prompt",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_stats()
        self.set_interval(30.0, self.refresh_stats)
        self.append_log("Workshop: Ctrl+D dashboard · Ctrl+E editor · Ctrl+G chat · Ctrl+H cycle.")
        self.append_log("Double-click .md files to preview. Prefix shell commands with ! or $.")
        self.query_one("#home-back", Button).display = False
        self.query_one("#home-prompt", Input).focus()

    def refresh_stats(self) -> None:
        app = self.app
        text = format_host_snapshot(app, compact=True)
        self.query_one("#home-stats", Static).update(text)

    def append_log(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            plain = _strip_markup(line)
            if self._log_lines and self._log_lines[-1] == plain:
                continue
            self._log_lines.append(plain)
        self._sync_log_text()

    def _sync_log_text(self) -> None:
        log = self.query_one("#home-log", TextArea)
        log.text = "\n".join(self._log_lines)
        log.scroll_end(animate=False)

    def action_copy_log(self) -> None:
        log = self.query_one("#home-log", TextArea)
        if not log.text.strip():
            self.notify("Nothing to copy.")
            return
        self.app.copy_to_clipboard(log.text)
        self.notify("Copied to clipboard.")

    def action_absorb_ctrl_c(self) -> None:
        self.notify("Use Ctrl+G for chat, Ctrl+D for dashboard.", timeout=2)

    async def show_markdown(self, path: Path) -> None:
        resolved = path.resolve()
        body = self.query_one("#home-content-body")
        log = self.query_one("#home-log", TextArea)
        log.display = False
        self.query_one("#home-copy", Button).display = False
        self.query_one("#home-back", Button).display = True
        self.query_one("#home-content-title", Static).update(resolved.name)
        if self._markdown_viewer is not None:
            await self._markdown_viewer.remove()
            self._markdown_viewer = None
        viewer = MarkdownViewer(show_table_of_contents=True, id="home-markdown")
        await body.mount(viewer)
        self._markdown_viewer = viewer
        self._view_mode = "markdown"
        try:
            await viewer.go(resolved)
        except Exception as exc:
            await self.show_log()
            self.append_log(f"Could not open {resolved}: {exc}")

    async def show_log(self) -> None:
        if self._markdown_viewer is not None:
            await self._markdown_viewer.remove()
            self._markdown_viewer = None
        self.query_one("#home-log", TextArea).display = True
        self.query_one("#home-copy", Button).display = True
        self.query_one("#home-back", Button).display = False
        self.query_one("#home-content-title", Static).update("Shell log")
        self._view_mode = "log"
        self._sync_log_text()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.handle_nav_button(event.button.id):
            return
        if event.button.id == "home-copy":
            self.action_copy_log()
        elif event.button.id == "home-back":
            self.run_worker(self.show_log, exclusive=True)

    def on_project_file_activated(self, event: ProjectFileActivated) -> None:
        path = event.path
        if path.suffix.lower() not in _MARKDOWN_SUFFIXES:
            return
        self.run_worker(self.show_markdown(path), exclusive=True)

    def action_leave_content_view(self) -> None:
        if self._view_mode == "markdown":
            self.run_worker(self.show_log, exclusive=True)
            return
        self.action_focus_prompt()

    def action_focus_prompt(self) -> None:
        self.query_one("#home-prompt", Input).focus()

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        app = self.app
        app.cwd = Path(event.path).resolve()
        self.append_log(f"[dim]cwd -> {app.cwd}[/dim]")
        self.refresh_stats()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        app = self.app
        raw = event.value.strip()
        event.input.value = ""
        if not raw:
            return
        if raw.lower() in ("chat", "/chat", "c"):
            self.run_worker(app.open_chat, exclusive=True)
            return
        if raw.lower() in ("dashboard", "/dashboard", "d"):
            self.run_worker(app.open_dashboard, exclusive=True)
            return
        if raw.lower() in ("/shell", "shell"):
            self.shell_mode = not self.shell_mode
            mode = "on" if self.shell_mode else "off"
            self.append_log(f"[dim]shell mode {mode}[/dim]")
            return
        shell_cmd = raw
        if raw.startswith(("!", "$")):
            shell_cmd = raw[1:].strip()
        elif not self.shell_mode:
            self.append_log("[dim]Prefix commands with ! or $, or type chat / dashboard.[/dim]")
            return
        if not shell_cmd:
            return
        self.append_log(f"[bold cyan]$[/bold cyan] {shell_cmd}")
        self.run_worker(lambda: app.run_shell_command(shell_cmd), thread=True, exclusive=True)
