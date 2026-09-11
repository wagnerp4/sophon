from __future__ import annotations

from typing import Any

from integrations.obsidian.client import (
    ObsidianClient,
    format_tool_payload,
    obsidian_tools_enabled,
)

VAULT_SEARCH = "vault_search"
VAULT_LIST = "vault_list"
VAULT_READ = "vault_read"
VAULT_RECENT = "vault_recent"

VAULT_TOOL_NAMES = frozenset({VAULT_SEARCH, VAULT_LIST, VAULT_READ, VAULT_RECENT})

VAULT_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": VAULT_SEARCH,
        "description": (
            "Full-text search the user's Obsidian vault via Local REST API. "
            "Use when the user asks about notes, vault content, or to find topics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (Obsidian simple search).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

VAULT_LIST_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": VAULT_LIST,
        "description": (
            "List files and folders in an Obsidian vault directory "
            "(relative to vault root). Empty path lists the vault root."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Vault-relative folder path (default empty = root).",
                },
            },
            "additionalProperties": False,
        },
    },
}

VAULT_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": VAULT_READ,
        "description": (
            "Read a single Obsidian note or text file by vault-relative path "
            "(e.g. Projects/notes.md)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Vault-relative file path.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

VAULT_RECENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": VAULT_RECENT,
        "description": (
            "List entries at the vault root (up to limit). "
            "Use as a quick overview when the user asks what is in the vault."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return (default 10, max 100).",
                },
            },
            "additionalProperties": False,
        },
    },
}

OBSIDIAN_TOOL_SYSTEM_HINT = (
    "You can read the user's Obsidian vault with vault_search, vault_list, "
    "vault_read, and vault_recent. Prefer vault_search to find notes, then "
    "vault_read for full content. Do not claim you cannot access the local vault "
    "when these tools are available. Do not invent file paths."
)


def obsidian_chat_tools() -> list[dict[str, Any]]:
    if not obsidian_tools_enabled():
        return []
    return [VAULT_SEARCH_TOOL, VAULT_LIST_TOOL, VAULT_READ_TOOL, VAULT_RECENT_TOOL]


def execute_vault_tool(name: str, arguments: dict[str, Any]) -> str:
    client = ObsidianClient()
    if name == VAULT_SEARCH:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "error: query is required"
        return format_tool_payload(client.search_simple(query))
    if name == VAULT_LIST:
        path = str(arguments.get("path") or "").strip()
        return format_tool_payload({"path": path or "/", "files": client.list_dir(path)})
    if name == VAULT_READ:
        path = str(arguments.get("path") or "").strip()
        if not path:
            return "error: path is required"
        return client.get_file(path)
    if name == VAULT_RECENT:
        limit = arguments.get("limit", 10)
        try:
            n = int(limit)
        except (TypeError, ValueError):
            n = 10
        return format_tool_payload({"files": client.recent_proxy(n)})
    return f"error: unknown vault tool {name!r}"
