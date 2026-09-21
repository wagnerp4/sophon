from __future__ import annotations

from dataclasses import dataclass

from textual.message import Message
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from integrations.google.drive import DriveEntry, list_children
from integrations.google.oauth import active_email, list_accounts


@dataclass
class DriveNode:
    kind: str
    file_id: str
    name: str
    mime_type: str = ""
    loaded: bool = False


class DriveItemActivated(Message):
    def __init__(self, file_id: str, name: str, is_folder: bool) -> None:
        self.file_id = file_id
        self.name = name
        self.is_folder = is_folder
        super().__init__()


class DriveTree(Tree[DriveNode]):
    def __init__(self) -> None:
        super().__init__("Drive", id="editor-drive-tree")
        self.guide_depth = 4

    def reload(self) -> None:
        self.clear()
        account = active_email()
        label = f"Drive · {account}" if account else "Drive"
        self.root.set_label(label)
        self.root.data = DriveNode("root", "root", "My Drive", loaded=False)
        if not list_accounts():
            self.root.add_leaf("(connect a Google account)")
            self.root.expand()
            return
        self.root.add_leaf("…")
        self.root.expand()

    def on_tree_node_expanded(self, event: Tree.NodeExpanded[DriveNode]) -> None:
        node = event.node
        data = node.data
        if data is None or data.loaded:
            return
        if data.kind not in ("root", "folder"):
            return
        data.loaded = True
        node.remove_children()
        try:
            children = list_children(data.file_id)
        except Exception as exc:
            node.add_leaf(f"(failed: {exc})")
            return
        if not children:
            node.add_leaf("(empty)")
            return
        for entry in children:
            self._add_entry(node, entry)

    def on_tree_node_selected(self, event: Tree.NodeSelected[DriveNode]) -> None:
        data = event.node.data
        if data is None:
            return
        if data.kind == "file":
            self.post_message(DriveItemActivated(data.file_id, data.name, False))
            return
        if data.kind == "folder":
            self.post_message(DriveItemActivated(data.file_id, data.name, True))

    def _add_entry(self, parent: TreeNode[DriveNode], entry: DriveEntry) -> None:
        if entry.is_folder:
            child = parent.add(
                entry.name,
                data=DriveNode("folder", entry.id, entry.name, entry.mime_type),
                allow_expand=True,
            )
            child.add_leaf("…")
            return
        parent.add_leaf(
            entry.name,
            data=DriveNode("file", entry.id, entry.name, entry.mime_type, loaded=True),
        )
