from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

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
    submenu: tuple["MenuItem", ...] = ()


@dataclass(frozen=True)
class _MenuRow:
    action: str
    payload: str
    prompt: str
    disabled: bool
    separator_before: bool
    toggle: bool


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


def _option_prompt(label: str, shortcut: str = "", width: int = 36) -> str:
    if not shortcut:
        return f" {label}"
    pad = width - len(label) - len(shortcut) - 2
    return f" {label}{' ' * max(1, pad)}{shortcut}"


class MenuPopup(ModalScreen[tuple[str, str] | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Close", show=False),
    ]
    _DROP_W: ClassVar[int] = 40
    _DROP_MAX_H: ClassVar[int] = 22

    DEFAULT_CSS = """
    MenuPopup {
        align: left top;
        background: transparent;
        layout: horizontal;
    }

    #editor-menu-dropdown {
        width: 40;
        height: auto;
        max-height: 22;
        border: solid $primary;
        background: $surface;
        padding: 0;
    }
    """

    def __init__(self, items: tuple[MenuItem, ...], origin: tuple[int, int]) -> None:
        super().__init__()
        self._items = items
        self._origin = origin
        self._expanded: set[str] = set()
        self._rows: list[_MenuRow] = []
        self._index_map: list[tuple[str, str]] = []

    def _flatten(self) -> list[_MenuRow]:
        rows: list[_MenuRow] = []
        for item in self._items:
            if item.submenu:
                opened = item.action in self._expanded
                mark = "▾" if opened else "▸"
                rows.append(
                    _MenuRow(
                        action=item.action,
                        payload="",
                        prompt=_option_prompt(f"{item.label}  {mark}"),
                        disabled=item.disabled,
                        separator_before=item.separator_before,
                        toggle=True,
                    )
                )
                if not opened:
                    continue
                for child in item.submenu:
                    rows.append(
                        _MenuRow(
                            action=child.action,
                            payload=child.payload,
                            prompt=_option_prompt(f"  {child.label}", child.shortcut),
                            disabled=child.disabled,
                            separator_before=child.separator_before,
                            toggle=False,
                        )
                    )
                continue
            rows.append(
                _MenuRow(
                    action=item.action,
                    payload=item.payload,
                    prompt=_option_prompt(item.label, item.shortcut),
                    disabled=item.disabled,
                    separator_before=item.separator_before,
                    toggle=False,
                )
            )
        return rows

    def _listing_content(self) -> tuple[list, list[_MenuRow]]:
        options: list = []
        rows = self._flatten()
        visible: list[_MenuRow] = []
        for row in rows:
            if row.separator_before:
                if Separator is not None:
                    options.append(Separator())
                else:
                    options.append(Option(" ──────────────", disabled=True))
                visible.append(
                    _MenuRow(
                        action="",
                        payload="",
                        prompt="",
                        disabled=True,
                        separator_before=True,
                        toggle=False,
                    )
                )
            options.append(
                Option(
                    row.prompt,
                    id=f"menu-item-{len(visible)}",
                    disabled=row.disabled,
                )
            )
            visible.append(row)
        return options, visible

    def _apply_rows(self, listing: OptionList, highlight: int | None = None) -> None:
        options, rows = self._listing_content()
        listing.clear_options()
        listing.add_options(options)
        self._rows = rows
        self._index_map = [(row.action, row.payload) for row in rows]
        self._place_dropdown(listing)
        if highlight is not None and rows:
            listing.highlighted = max(0, min(highlight, len(rows) - 1))

    def _place_dropdown(self, listing: OptionList) -> None:
        count = max(len(self._rows), 1)
        drop_h = min(count + 2, self._DROP_MAX_H)
        listing.styles.height = drop_h
        screen_w = self.size.width or (self.app.size.width if self.app.size else 80)
        screen_h = self.size.height or (self.app.size.height if self.app.size else 24)
        drop_w = self._DROP_W
        x, y = self._origin
        if x + drop_w > screen_w:
            x = max(0, screen_w - drop_w)
        if y + drop_h > screen_h:
            y = max(0, self._origin[1] - drop_h - 1)
        listing.styles.offset = (max(0, x), max(0, y))

    def compose(self) -> ComposeResult:
        options, rows = self._listing_content()
        self._rows = rows
        self._index_map = [(row.action, row.payload) for row in rows]
        yield OptionList(*options, id="editor-menu-dropdown")

    def on_mount(self) -> None:
        listing = self.query_one("#editor-menu-dropdown", OptionList)
        self._place_dropdown(listing)
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
        if index < 0 or index >= len(self._rows):
            self.dismiss(None)
            return
        row = self._rows[index]
        if row.toggle:
            event.stop()
            if row.action in self._expanded:
                self._expanded.discard(row.action)
            else:
                self._expanded.add(row.action)
            listing = self.query_one("#editor-menu-dropdown", OptionList)
            self._apply_rows(listing, highlight=index)
            listing.focus()
            return
        if not row.action:
            return
        self.dismiss((row.action, row.payload))


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
        # TODO: hover-switch between open menus, Alt/F10 keyboard access, flyout nested submenus

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
