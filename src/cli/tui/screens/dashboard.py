from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Static

from cli.tui.footer import NexusFooter
from cli.tui.screens.nav import NAV_BINDINGS, ModeNavigationMixin, compose_nav_bar
from cli.tui.tiles.base import TileState
from cli.tui.tiles.registry import build_dashboard_tiles

_DASH_LAYOUTS = ("auto", "fill", "scroll", "narrow")
_DASH_LAYOUT_CLASSES = ("-dash-fill", "-dash-scroll", "-dash-narrow")


class DashboardScreen(ModeNavigationMixin, Screen):
    BINDINGS = [
        *NAV_BINDINGS,
        Binding("f4", "cycle_dash_layout", "Layout", show=True),
        Binding("ctrl+c", "absorb_ctrl_c", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._tiles, self._unknown = build_dashboard_tiles()
        self._states: dict[str, TileState] = {}
        self._tile_timers: list = []
        self._panels: dict[str, object] = {}
        self._active = False
        self._layout_mode = "auto"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield from compose_nav_bar("dashboard")
        with Vertical(id="dash-root"):
            with Grid(id="dash-grid"):
                for tile in self._tiles:
                    panel = tile.panel()
                    self._panels[tile.tile_id] = panel
                    yield panel
            yield Static("", id="dash-notes")
        yield NexusFooter()

    def on_mount(self) -> None:
        self._active = True
        notes = self.query_one("#dash-notes", Static)
        if self._unknown:
            notes.update(f"[dim]unknown tiles skipped: {', '.join(self._unknown)}[/dim]")
        else:
            notes.update("[dim]Ctrl+E editor · Ctrl+G chat · Ctrl+H cycle · F4 layout[/dim]")
        self._apply_dash_layout()
        for tile in self._tiles:
            self._schedule_tile(tile)

    def on_resize(self, _event: events.Resize) -> None:
        self._apply_dash_layout()

    def _resolved_dash_layout(self) -> str:
        if self._layout_mode != "auto":
            return self._layout_mode
        size = self.size
        if size.width < 92:
            return "narrow"
        if size.height < 38:
            return "scroll"
        return "fill"

    def _apply_dash_layout(self) -> None:
        resolved = self._resolved_dash_layout()
        for css_class in _DASH_LAYOUT_CLASSES:
            self.remove_class(css_class)
        self.add_class(f"-dash-{resolved}")

    def action_cycle_dash_layout(self) -> None:
        index = _DASH_LAYOUTS.index(self._layout_mode)
        self._layout_mode = _DASH_LAYOUTS[(index + 1) % len(_DASH_LAYOUTS)]
        self._apply_dash_layout()
        resolved = self._resolved_dash_layout()
        if self._layout_mode == "auto":
            self.notify(f"dashboard layout auto ({resolved})")
            return
        self.notify(f"dashboard layout {resolved}")

    def on_unmount(self) -> None:
        self._active = False
        self._pause_timers()
        self._tile_timers.clear()

    def on_show(self) -> None:
        self._active = True
        for timer in self._tile_timers:
            try:
                timer.resume()
            except Exception:
                pass
        for tile in self._tiles:
            self.refresh_tile(tile.tile_id)

    def on_hide(self) -> None:
        self._active = False
        self._pause_timers()

    def _pause_timers(self) -> None:
        for timer in self._tile_timers:
            try:
                timer.pause()
            except Exception:
                pass

    def _schedule_tile(self, tile) -> None:
        self.refresh_tile(tile.tile_id)
        timer = self.set_interval(
            tile.refresh_s,
            lambda t=tile: self.refresh_tile(t.tile_id),
            pause=False,
        )
        self._tile_timers.append(timer)

    def refresh_tile(self, tile_id: str) -> None:
        if not self._active:
            return
        tile = next((item for item in self._tiles if item.tile_id == tile_id), None)
        if tile is None:
            return
        self.run_worker(lambda: self._fetch_worker(tile), thread=True, exclusive=False)

    def _fetch_worker(self, tile) -> None:
        app = self.app
        state = tile.fetch(app)
        app.call_from_thread(self._apply_state, tile.tile_id, state)

    def _apply_state(self, tile_id: str, state: TileState) -> None:
        if not self._active:
            return
        self._states[tile_id] = state
        panel = self._panels.get(tile_id)
        if panel is None:
            try:
                panel = self.query_one(f"#dash-{tile_id}")
            except Exception:
                return
        apply = getattr(panel, "apply", None)
        if callable(apply):
            apply(state)
            return
        if isinstance(panel, Static):
            panel.update(state.text)

    def refresh_stats(self) -> None:
        self.refresh_tile("host")
        self.refresh_tile("system")

    def action_absorb_ctrl_c(self) -> None:
        self.notify("Use Ctrl+E for editor, Ctrl+G for chat.", timeout=2)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.handle_nav_button(event.button.id)
