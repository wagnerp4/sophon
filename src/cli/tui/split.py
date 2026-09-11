from __future__ import annotations

from textual import events
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static


class PaneSplitter(Static):
    DEFAULT_CSS = """
    PaneSplitter {
        background: $secondary;
        color: $text-muted;
        text-align: center;
    }

    PaneSplitter:hover, PaneSplitter.-dragging {
        background: $accent;
        color: $foreground;
    }

    PaneSplitter.-axis-x {
        width: 1;
        height: 1fr;
        min-width: 1;
        pointer: ew-resize;
    }

    PaneSplitter.-axis-y {
        width: 1fr;
        height: 1;
        min-height: 1;
        pointer: ns-resize;
    }
    """

    class CollapseRequested(Message):
        def __init__(self, splitter: PaneSplitter) -> None:
            self.splitter = splitter
            super().__init__()

    def __init__(
        self,
        *,
        target_id: str,
        axis: str = "x",
        invert: bool = False,
        min_size: int = 16,
        max_size: int | None = None,
        flex_min: int = 24,
        companion_ids: tuple[str, ...] = (),
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        glyph = "┊" if axis == "x" else "┄"
        super().__init__(glyph, name=name, id=id, classes=classes, disabled=disabled, markup=False)
        self.target_id = target_id
        self.axis = axis
        self.invert = invert
        self.min_size = min_size
        self.max_size = max_size
        self.flex_min = flex_min
        self.companion_ids = companion_ids
        self._dragging = False
        self._origin = 0
        self._origin_size = 0
        self.add_class("-axis-x" if axis == "x" else "-axis-y")

    def _target(self) -> Widget | None:
        try:
            return self.screen.query_one(f"#{self.target_id}")
        except NoMatches:
            return None

    def _coord(self, event: events.MouseEvent) -> int:
        if self.axis == "x":
            return event.screen_x
        return event.screen_y

    def _read_size(self, target: Widget) -> int:
        if self.axis == "x":
            return target.size.width
        return target.size.height

    def _write_size(self, target: Widget, size: int) -> None:
        if self.axis == "x":
            target.styles.width = size
        else:
            target.styles.height = size
        if self.axis != "x":
            return
        for companion_id in self.companion_ids:
            try:
                companion = self.screen.query_one(f"#{companion_id}")
            except NoMatches:
                continue
            companion.styles.width = size

    def _available_max(self) -> int:
        parent = self.parent
        target = self._target()
        if not isinstance(parent, Widget) or target is None:
            return self.max_size or 120
        used = 0
        if self.axis == "x":
            parent_span = parent.size.width
            splitter_span = max(self.size.width, 1)
        else:
            parent_span = parent.size.height
            splitter_span = max(self.size.height, 1)
        used += splitter_span
        for child in parent.children:
            if child is self or child is target:
                continue
            if not child.display:
                continue
            if "editor-flex" in child.classes:
                used += self.flex_min
                continue
            used += child.size.width if self.axis == "x" else child.size.height
        budget = max(self.min_size, parent_span - used)
        if self.max_size is not None:
            return min(self.max_size, budget)
        return budget

    def _apply(self, size: int) -> None:
        target = self._target()
        if target is None:
            return
        clamped = max(self.min_size, min(size, self._available_max()))
        self._write_size(target, clamped)

    def _stop_drag(self) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self.remove_class("-dragging")
        self.release_mouse()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        event.stop()
        target = self._target()
        if target is None or not target.display:
            return
        self._dragging = True
        self._origin = self._coord(event)
        self._origin_size = self._read_size(target)
        self._write_size(target, self._origin_size)
        self.add_class("-dragging")
        self.capture_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self._dragging:
            return
        event.stop()
        delta = self._coord(event) - self._origin
        if self.invert:
            delta = -delta
        self._apply(self._origin_size + delta)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._stop_drag()
        event.stop()

    def on_mouse_release(self, event: events.MouseRelease) -> None:
        self._stop_drag()
        event.stop()

    def on_click(self, event: events.Click) -> None:
        if event.chain != 2:
            return
        event.stop()
        self._stop_drag()
        self.post_message(self.CollapseRequested(self))
