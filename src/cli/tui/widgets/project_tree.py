from __future__ import annotations

from pathlib import Path
from typing import Iterable

from textual import events
from textual.message import Message
from textual.widgets import DirectoryTree
from textual.widgets.directory_tree import DirEntry


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
