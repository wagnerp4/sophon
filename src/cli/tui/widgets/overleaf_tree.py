from __future__ import annotations

from dataclasses import dataclass

from textual.message import Message
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from integrations.overleaf.client import (
    OverleafEntry,
    ensure_synced,
    list_files,
    list_projects,
    overleaf_available,
)


@dataclass
class OverleafNode:
    kind: str
    project_id: str
    relpath: str
    name: str
    loaded: bool = False


class OverleafItemActivated(Message):
    def __init__(self, project_id: str, relpath: str, name: str, is_dir: bool) -> None:
        self.project_id = project_id
        self.relpath = relpath
        self.name = name
        self.is_dir = is_dir
        super().__init__()


class OverleafTree(Tree[OverleafNode]):
    def __init__(self) -> None:
        super().__init__("Overleaf", id="editor-overleaf-tree")
        self.guide_depth = 4

    def reload(self) -> None:
        self.clear()
        self.root.set_label("Overleaf")
        self.root.data = OverleafNode("root", "", "", "Overleaf", loaded=True)
        if not overleaf_available():
            self.root.add_leaf("(set SOPHON_OVERLEAF_GIT_TOKEN and PROJECT_ID)")
            self.root.expand()
            return
        projects = list_projects(sync=False)
        if not projects:
            self.root.add_leaf("(no projects configured)")
            self.root.expand()
            return
        for project in projects:
            label = project.alias
            if project.alias != project.project_id:
                label = f"{project.alias}  ({project.project_id})"
            child = self.root.add(
                label,
                data=OverleafNode("project", project.project_id, "", project.alias),
                allow_expand=True,
            )
            child.add_leaf("…")
        self.root.expand()

    def on_tree_node_expanded(self, event: Tree.NodeExpanded[OverleafNode]) -> None:
        node = event.node
        data = node.data
        if data is None or data.loaded:
            return
        if data.kind not in ("project", "folder"):
            return
        data.loaded = True
        node.remove_children()
        try:
            if data.kind == "project":
                ensure_synced(data.project_id)
            entries = list_files(data.project_id, data.relpath)
        except Exception as exc:
            node.add_leaf(f"(failed: {exc})")
            return
        if not entries:
            node.add_leaf("(empty)")
            return
        for entry in entries:
            self._add_entry(node, data.project_id, entry)

    def on_tree_node_selected(self, event: Tree.NodeSelected[OverleafNode]) -> None:
        data = event.node.data
        if data is None:
            return
        if data.kind == "file":
            self.post_message(
                OverleafItemActivated(data.project_id, data.relpath, data.name, False)
            )
            return
        if data.kind == "folder":
            self.post_message(
                OverleafItemActivated(data.project_id, data.relpath, data.name, True)
            )

    def _add_entry(
        self,
        parent: TreeNode[OverleafNode],
        project_id: str,
        entry: OverleafEntry,
    ) -> None:
        if entry.is_dir:
            child = parent.add(
                entry.name,
                data=OverleafNode("folder", project_id, entry.relpath, entry.name),
                allow_expand=True,
            )
            child.add_leaf("…")
            return
        parent.add_leaf(
            entry.name,
            data=OverleafNode(
                "file",
                project_id,
                entry.relpath,
                entry.name,
                loaded=True,
            ),
        )
