from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

try:
    from textual.widgets.option_list import Separator
except ImportError:
    Separator = None


@dataclass(frozen=True)
class MenuItem:
    label: str
    action: str
    shortcut: str = ""
    disabled: bool = False
    separator_before: bool = False
    payload: str = ""


@dataclass(frozen=True)
class MenuGroup:
    key: str
    label: str
    items: tuple[MenuItem, ...]


class MenuAction(Message):
    def __init__(self, action: str, pane: str, payload: str = "") -> None:
        self.action = action
        self.pane = pane
        self.payload = payload
        super().__init__()


def _option_prompt(item: MenuItem, width: int = 30) -> str:
    label = item.label
    shortcut = item.shortcut
    if not shortcut:
        return f" {label}"
    pad = width - len(label) - len(shortcut) - 2
    return f" {label}{' ' * max(1, pad)}{shortcut}"


class MenuPopup(ModalScreen[tuple[str, str] | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Close", show=False),
    ]

    DEFAULT_CSS = """
    MenuPopup {
        align: left top;
        background: transparent;
        layout: horizontal;
    }

    #editor-menu-dropdown {
        width: 34;
        height: auto;
        max-height: 18;
        border: solid $primary;
        background: $surface;
        padding: 0;
    }
    """

    def __init__(self, items: tuple[MenuItem, ...], origin: tuple[int, int]) -> None:
        super().__init__()
        self._items = items
        self._origin = origin
        self._index_map: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        options: list = []
        self._index_map = []
        for item in self._items:
            if item.separator_before:
                if Separator is not None:
                    options.append(Separator())
                else:
                    options.append(Option(" ──────────────", disabled=True))
                self._index_map.append(("", ""))
            options.append(
                Option(
                    _option_prompt(item),
                    id=f"menu-item-{len(self._index_map)}",
                    disabled=item.disabled,
                )
            )
            self._index_map.append((item.action, item.payload))
        yield OptionList(*options, id="editor-menu-dropdown")

    def on_mount(self) -> None:
        listing = self.query_one("#editor-menu-dropdown", OptionList)
        count = max(len(self._index_map), 1)
        listing.styles.height = min(count + 2, 18)
        screen_w = self.size.width or (self.app.size.width if self.app.size else 80)
        screen_h = self.size.height or (self.app.size.height if self.app.size else 24)
        drop_w = 34
        drop_h = min(count + 2, 18)
        x, y = self._origin
        if x + drop_w > screen_w:
            x = max(0, screen_w - drop_w)
        if y + drop_h > screen_h:
            y = max(0, self._origin[1] - drop_h - 1)
        listing.styles.offset = (max(0, x), max(0, y))
        listing.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        listing = self.query_one("#editor-menu-dropdown", OptionList)
        region = listing.region
        inside = (
            region.x <= event.screen_x < region.x + region.width
            and region.y <= event.screen_y < region.y + region.height
        )
        if inside:
            return
        event.stop()
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        if index < 0 or index >= len(self._index_map):
            self.dismiss(None)
            return
        action, payload = self._index_map[index]
        if not action:
            return
        self.dismiss((action, payload))


class PaneMenuBar(Horizontal):
    DEFAULT_CSS = """
    PaneMenuBar {
        height: 1;
        width: 1fr;
        layout: horizontal;
        background: $surface;
    }

    PaneMenuBar > .editor-menubar-label {
        width: 1fr;
        min-width: 8;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    PaneMenuBar > .editor-menubar-item {
        min-width: 6;
        height: 1;
        border: none;
        padding: 0 1;
        background: $surface;
    }

    PaneMenuBar > .editor-menubar-item:hover,
    PaneMenuBar > .editor-menubar-item.-open {
        background: $primary;
        color: $text;
    }
    """

    def __init__(
        self,
        pane: str,
        title: str,
        menus: Callable[[], tuple[MenuGroup, ...]],
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.pane = pane
        self._title = title
        self._menus = menus
        self._open_button: Button | None = None
        # TODO: hover-switch between open menus, Alt/F10 keyboard access, nested submenus

    def compose(self) -> ComposeResult:
        for group in self._menus():
            yield Button(
                group.label,
                id=f"editor-menu-{self.pane}-{group.key}",
                classes="editor-menubar-item",
            )
        yield Static(self._title, classes="editor-menubar-label")

    def set_title(self, text: str) -> None:
        self._title = text
        try:
            self.query_one(".editor-menubar-label", Static).update(text)
        except NoMatches:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button
        button_id = button.id or ""
        prefix = f"editor-menu-{self.pane}-"
        if not button_id.startswith(prefix):
            return
        key = button_id[len(prefix) :]
        group = next((item for item in self._menus() if item.key == key), None)
        if group is None or not group.items:
            return
        region = button.region
        origin = (region.x, region.y + region.height)
        button.add_class("-open")
        self._open_button = button
        self.app.push_screen(MenuPopup(group.items, origin), self._on_popup_closed)

    def _on_popup_closed(self, result: tuple[str, str] | None) -> None:
        if self._open_button is not None:
            self._open_button.remove_class("-open")
            self._open_button = None
        if result is None:
            return
        action, payload = result
        if not action:
            return
        self.post_message(MenuAction(action, self.pane, payload))
