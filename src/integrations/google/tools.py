from __future__ import annotations

import json
from typing import Any

from integrations.google.bookmarks import bookmarks_available, format_bookmarks_tree
from integrations.google.drive import format_tree as drive_format_tree
from integrations.google.drive import list_children, read_file_text
from integrations.google.gmail import format_search_results, read_message, search_messages
from integrations.google.oauth import google_tools_enabled, list_accounts
from integrations.google.search import (
    format_search_results as format_web_results,
)
from integrations.google.search import search_web, web_search_tools_enabled

GMAIL_SEARCH = "gmail_search"
GMAIL_READ = "gmail_read"
DRIVE_TREE = "drive_tree"
DRIVE_LIST = "drive_list"
BOOKMARKS_TREE = "bookmarks_tree"
WEB_SEARCH = "web_search"

GOOGLE_TOOL_NAMES = frozenset(
    {GMAIL_SEARCH, GMAIL_READ, DRIVE_TREE, DRIVE_LIST, BOOKMARKS_TREE, WEB_SEARCH}
)

GMAIL_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GMAIL_SEARCH,
        "description": (
            "Search the user's Gmail with a Gmail query string (from:, subject:, newer_than:, etc.). "
            "Use when they ask to find mail. Returns ids, subjects, senders, snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Gmail search query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max messages (default 10, max 25).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

GMAIL_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GMAIL_READ,
        "description": (
            "Read one Gmail message by id from gmail_search. Returns subject, headers, and body text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Gmail message id.",
                },
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    },
}

DRIVE_TREE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DRIVE_TREE,
        "description": (
            "List Google Drive folders/files as a shallow tree under a parent id "
            "(default root). Use for Drive structure, not for downloading binaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "parent_id": {
                    "type": "string",
                    "description": "Drive folder id. Empty = My Drive root.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Folder levels to expand (default 2, max 6).",
                },
            },
            "additionalProperties": False,
        },
    },
}

DRIVE_LIST_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DRIVE_LIST,
        "description": (
            "List children of one Google Drive folder id, or read text metadata for a file id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Drive file or folder id (default root).",
                },
                "read": {
                    "type": "boolean",
                    "description": "If true, return text content / export for a file.",
                },
            },
            "additionalProperties": False,
        },
    },
}

BOOKMARKS_TREE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": BOOKMARKS_TREE,
        "description": (
            "List the user's Chrome bookmark export as a folder tree "
            "(SOPHON_BOOKMARKS_PATH Netscape HTML)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "depth": {
                    "type": "integer",
                    "description": "Folder levels to expand (default 3, max 8).",
                },
            },
            "additionalProperties": False,
        },
    },
}

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": WEB_SEARCH,
        "description": (
            "Search the public web. Uses SearXNG when SOPHON_SEARXNG_URL is set, "
            "otherwise Google Custom Search. Returns titles, urls, and snippets. Not personalized."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search phrase.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max hits (default 5, max 10).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

GOOGLE_TOOL_SYSTEM_HINT = (
    "You can use gmail_search / gmail_read for Gmail, drive_tree / drive_list for Google Drive, "
    "bookmarks_tree for the Chrome bookmark export, and web_search for live web results "
    "(SearXNG or Google Custom Search). "
    "Do not claim you cannot access these when the tools are listed. "
    "TUM / Outlook mail is not available. Prefer search tools over guessing."
)


def google_chat_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if google_tools_enabled() and list_accounts():
        tools.extend(
            [
                GMAIL_SEARCH_TOOL,
                GMAIL_READ_TOOL,
                DRIVE_TREE_TOOL,
                DRIVE_LIST_TOOL,
            ]
        )
    if bookmarks_available():
        tools.append(BOOKMARKS_TREE_TOOL)
    if web_search_tools_enabled():
        tools.append(WEB_SEARCH_TOOL)
    return tools


def execute_google_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == GMAIL_SEARCH:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "error: query is required"
        limit = arguments.get("limit", 10)
        try:
            n = int(limit)
        except (TypeError, ValueError):
            n = 10
        messages = search_messages(query, limit=n)
        return _payload(format_search_results(messages, query=query))
    if name == GMAIL_READ:
        mid = str(arguments.get("id") or "").strip()
        if not mid:
            return "error: id is required"
        return _payload(read_message(mid))
    if name == DRIVE_TREE:
        parent = str(arguments.get("parent_id") or "root").strip() or "root"
        depth = arguments.get("depth", 2)
        try:
            levels = int(depth)
        except (TypeError, ValueError):
            levels = 2
        return _payload(drive_format_tree(parent, depth=levels))
    if name == DRIVE_LIST:
        fid = str(arguments.get("id") or "root").strip() or "root"
        if bool(arguments.get("read")):
            return _payload(read_file_text(fid))
        children = list_children(fid)
        return _payload(
            {
                "id": fid,
                "count": len(children),
                "children": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "mime_type": c.mime_type,
                        "is_folder": c.is_folder,
                        "web_view_link": c.web_view_link,
                    }
                    for c in children
                ],
            }
        )
    if name == BOOKMARKS_TREE:
        depth = arguments.get("depth", 3)
        try:
            levels = int(depth)
        except (TypeError, ValueError):
            levels = 3
        return _payload(format_bookmarks_tree(depth=levels))
    if name == WEB_SEARCH:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "error: query is required"
        limit = arguments.get("limit", 5)
        try:
            n = int(limit)
        except (TypeError, ValueError):
            n = 5
        hits = search_web(query, limit=n)
        return _payload(format_web_results(hits, query=query))
    return f"error: unknown google tool {name!r}"


def _payload(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
