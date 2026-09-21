from __future__ import annotations

from dataclasses import dataclass

from textual.message import Message
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from integrations.google.bookmarks import BookmarkNode, bookmarks_available, load_bookmarks


@dataclass
class BookmarkTreeNode:
    kind: str
    title: str
    url: str = ""
    loaded: bool = False
    children: list[BookmarkNode] | None = None


class BookmarkActivated(Message):
    def __init__(self, title: str, url: str) -> None:
        self.title = title
        self.url = url
        super().__init__()


class BookmarksTree(Tree[BookmarkTreeNode]):
    def __init__(self) -> None:
        super().__init__("Bookmarks", id="editor-bookmarks-tree")
        self.guide_depth = 4
        self._root_model: BookmarkNode | None = None

    def reload(self) -> None:
        self.clear()
        if not bookmarks_available():
            self.root.set_label("Bookmarks")
            self.root.data = BookmarkTreeNode("root", "Bookmarks", loaded=True)
            self.root.add_leaf("(set SOPHON_BOOKMARKS_PATH)")
            self.root.expand()
            return
        try:
            self._root_model = load_bookmarks()
        except Exception as exc:
            self.root.set_label("Bookmarks")
            self.root.data = BookmarkTreeNode("root", "Bookmarks", loaded=True)
            self.root.add_leaf(f"(unavailable: {exc})")
            self.root.expand()
            return
        self.root.set_label(self._root_model.title)
        self.root.data = BookmarkTreeNode(
            "folder",
            self._root_model.title,
            loaded=False,
            children=list(self._root_model.children),
        )
        self.root.add_leaf("…")
        self.root.expand()

    def on_tree_node_expanded(self, event: Tree.NodeExpanded[BookmarkTreeNode]) -> None:
        node = event.node
        data = node.data
        if data is None or data.loaded or data.kind != "folder":
            return
        data.loaded = True
        node.remove_children()
        children = data.children or []
        if not children:
            node.add_leaf("(empty)")
            return
        for child in children:
            self._add_child(node, child)

    def on_tree_node_selected(self, event: Tree.NodeSelected[BookmarkTreeNode]) -> None:
        data = event.node.data
        if data is None or data.kind != "bookmark" or not data.url:
            return
        self.post_message(BookmarkActivated(data.title, data.url))

    def _add_child(self, parent: TreeNode[BookmarkTreeNode], child: BookmarkNode) -> None:
        if child.kind == "folder":
            node = parent.add(
                child.title,
                data=BookmarkTreeNode(
                    "folder",
                    child.title,
                    loaded=False,
                    children=list(child.children),
                ),
                allow_expand=True,
            )
            node.add_leaf("…")
            return
        parent.add_leaf(
            child.title,
            data=BookmarkTreeNode("bookmark", child.title, url=child.url, loaded=True),
        )
