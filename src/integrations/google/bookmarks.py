from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from integrations.google.paths import open_path_exists, resolve_user_path


@dataclass
class BookmarkNode:
    kind: str
    title: str
    url: str = ""
    children: list[BookmarkNode] = field(default_factory=list)


class _NetscapeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = BookmarkNode("folder", "Bookmarks")
        self._stack: list[BookmarkNode] = [self.root]
        self._pending_folder: str | None = None
        self._pending_link: dict[str, str] | None = None
        self._capture: str = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        attr = {k.lower(): (v or "") for k, v in attrs}
        if name == "h3":
            self._pending_folder = ""
            self._capture = "folder"
            return
        if name == "a":
            self._pending_link = {
                "url": attr.get("href", ""),
                "title": "",
            }
            self._capture = "link"
            return
        if name == "dl":
            if self._pending_folder is not None:
                folder = BookmarkNode("folder", self._pending_folder or "Folder")
                self._stack[-1].children.append(folder)
                self._stack.append(folder)
                self._pending_folder = None
                self._capture = ""

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name == "h3" and self._capture == "folder":
            self._capture = ""
            return
        if name == "a" and self._pending_link is not None:
            title = (self._pending_link.get("title") or "").strip() or self._pending_link.get("url", "")
            url = (self._pending_link.get("url") or "").strip()
            if url and not url.lower().startswith("javascript:"):
                self._stack[-1].children.append(BookmarkNode("bookmark", title, url=url))
            self._pending_link = None
            self._capture = ""
            return
        if name == "dl":
            if len(self._stack) > 1:
                self._stack.pop()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._capture == "folder" and self._pending_folder is not None:
            self._pending_folder = f"{self._pending_folder}{text}".strip()
        elif self._capture == "link" and self._pending_link is not None:
            prev = self._pending_link.get("title", "")
            self._pending_link["title"] = f"{prev}{text}".strip()


def bookmarks_path() -> Path | None:
    raw = os.environ.get("SOPHON_BOOKMARKS_PATH", "").strip()
    if not raw:
        return None
    path = resolve_user_path(raw)
    if path is None:
        return None
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(
            path.glob("bookmarks_*.html"),
            key=lambda p: p.stat().st_mtime if p.is_file() else 0.0,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        html_files = sorted(
            path.glob("*.html"),
            key=lambda p: p.stat().st_mtime if p.is_file() else 0.0,
            reverse=True,
        )
        return html_files[0] if html_files else None
    return None


def bookmarks_available() -> bool:
    return open_path_exists(bookmarks_path())


def load_bookmarks(path: Path | None = None) -> BookmarkNode:
    target = path or bookmarks_path()
    if target is None or not target.is_file():
        raise RuntimeError(
            "Set SOPHON_BOOKMARKS_PATH to a Netscape bookmarks HTML file or export folder."
        )
    text = target.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r'\sICON="data:image/[^"]*"', "", text, flags=re.IGNORECASE)
    parser = _NetscapeParser()
    parser.feed(text)
    root = parser.root
    root.title = f"Bookmarks · {target.name}"
    return root


def format_bookmarks_tree(node: BookmarkNode | None = None, *, depth: int = 3) -> dict[str, Any]:
    root = node or load_bookmarks()
    levels = max(1, min(int(depth), 8))

    def _walk(item: BookmarkNode, remaining: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": item.kind,
            "title": item.title,
        }
        if item.kind == "bookmark":
            payload["url"] = item.url
            return payload
        if remaining <= 1:
            payload["child_count"] = len(item.children)
            return payload
        payload["children"] = [_walk(child, remaining - 1) for child in item.children]
        return payload

    return {
        "path": str(bookmarks_path()) if bookmarks_path() else None,
        "tree": _walk(root, levels),
    }


def bookmark_to_markdown(title: str, url: str) -> str:
    safe_title = html.escape(title or url or "Bookmark")
    return f"# {safe_title}\n\n{url}\n"


def find_folder(root: BookmarkNode, name: str) -> BookmarkNode | None:
    needle = (name or "").strip().lower()
    if not needle:
        return root
    if root.title.lower() == needle:
        return root
    for child in root.children:
        if child.kind != "folder":
            continue
        if child.title.lower() == needle:
            return child
        found = find_folder(child, name)
        if found is not None:
            return found
    return None
