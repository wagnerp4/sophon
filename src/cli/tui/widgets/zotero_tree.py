from __future__ import annotations

from dataclasses import dataclass

from textual.message import Message
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from integrations.zotero.client import ZoteroClient, ZoteroCollection, ZoteroItem


@dataclass
class ZoteroNode:
    kind: str
    key: str
    title: str
    item_type: str = ""
    loaded: bool = False


class ZoteroItemActivated(Message):
    def __init__(self, key: str, title: str) -> None:
        self.key = key
        self.title = title
        super().__init__()


class ZoteroTree(Tree[ZoteroNode]):
    def __init__(self) -> None:
        super().__init__("Zotero", id="editor-zotero-tree")
        self.guide_depth = 4
        self._client = ZoteroClient()
        self._collections: list[ZoteroCollection] = []
        self._by_parent: dict[str | None, list[ZoteroCollection]] = {}

    def reload(self) -> None:
        self.clear()
        self.root.set_label("Zotero")
        self.root.data = ZoteroNode("library", "", "Zotero", loaded=True)
        # TODO: collection search box, saved searches, tags root
        try:
            self._collections = self._client.list_collections()
        except Exception as exc:
            self.root.add_leaf(f"(unavailable: {exc})")
            self.root.expand()
            return
        grouped: dict[str | None, list[ZoteroCollection]] = {}
        for col in self._collections:
            grouped.setdefault(col.parent_key, []).append(col)
        for rows in grouped.values():
            rows.sort(key=lambda item: item.name.lower())
        self._by_parent = grouped
        tops = grouped.get(None, [])
        if not tops:
            self.root.add_leaf("(empty library)")
            self.root.expand()
            return
        for col in tops:
            self._add_collection(self.root, col)
        self.root.expand()

    def _add_collection(self, parent: TreeNode[ZoteroNode], col: ZoteroCollection) -> None:
        label = col.name
        if col.n_items:
            label = f"{col.name}  ({col.n_items})"
        node = parent.add(
            label,
            data=ZoteroNode("collection", col.key, col.name),
            allow_expand=True,
        )
        node.add_leaf("…")

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        data = node.data
        if data is None or data.kind != "collection" or data.loaded:
            return
        data.loaded = True
        node.remove_children()
        for col in self._by_parent.get(data.key, []):
            self._add_collection(node, col)
        try:
            items = self._client.list_collection_items(data.key)
        except Exception as exc:
            node.add_leaf(f"(items failed: {exc})")
            return
        if not items and not self._by_parent.get(data.key):
            node.add_leaf("(empty)")
            return
        for item in items:
            node.add_leaf(self._item_label(item), data=ZoteroNode("item", item.key, item.title, item.item_type))

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if data is None or data.kind != "item":
            return
        self.post_message(ZoteroItemActivated(data.key, data.title))

    def _item_label(self, item: ZoteroItem) -> str:
        bits = [item.title]
        extra: list[str] = []
        if item.year:
            extra.append(item.year)
        if item.creators:
            extra.append(item.creators)
        if extra:
            bits.append(f"  [{', '.join(extra)}]")
        return "".join(bits)
