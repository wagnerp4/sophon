from __future__ import annotations

from typing import Any

from integrations.zotero.client import (
    ZoteroClient,
    format_tool_payload,
    zotero_tools_enabled,
)

ZOTERO_TREE = "zotero_tree"
ZOTERO_SEARCH = "zotero_search"
ZOTERO_LIST = "zotero_list"
ZOTERO_READ = "zotero_read"
ZOTERO_METRICS = "zotero_metrics"

ZOTERO_TOOL_NAMES = frozenset(
    {ZOTERO_TREE, ZOTERO_SEARCH, ZOTERO_LIST, ZOTERO_READ, ZOTERO_METRICS}
)

ZOTERO_TREE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ZOTERO_TREE,
        "description": (
            "List the user's Zotero collection tree (folders). "
            "Use this when they ask about their zotero tree, library structure, or collections. "
            "Do not use shell_ls for Zotero."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name, path (Computer Science/NLP), or 8-char key. Empty = library root.",
                },
                "depth": {
                    "type": "integer",
                    "description": "How many folder levels to expand (default 2, max 8).",
                },
            },
            "additionalProperties": False,
        },
    },
}

ZOTERO_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ZOTERO_SEARCH,
        "description": (
            "Search the Zotero library by title, authors, citation key, and indexed PDF text. "
            "Use when the user asks for papers, a term, a topic, or 'query my zotero tree'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search phrase (title, author, citation key, or PDF term).",
                },
                "collection": {
                    "type": "string",
                    "description": "Optional collection name/path/key to scope the search.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max hits (default 20, max 50).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

ZOTERO_LIST_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ZOTERO_LIST,
        "description": (
            "List papers in one Zotero collection. "
            "Use after zotero_tree when the user wants the documents in a folder. "
            "Do not dump the entire library."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name, path, or 8-char key.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max items (default 40, max 100).",
                },
            },
            "required": ["collection"],
            "additionalProperties": False,
        },
    },
}

ZOTERO_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ZOTERO_READ,
        "description": (
            "Read one Zotero item: metadata, abstract, and optional PDF text excerpt. "
            "Pass the 8-char key from search/list, or a title. "
            "Set include_pdf_text true only when the user wants the paper body."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Item key or title to resolve.",
                },
                "include_pdf_text": {
                    "type": "boolean",
                    "description": "If true, append a truncated local PDF excerpt.",
                },
            },
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}

ZOTERO_METRICS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ZOTERO_METRICS,
        "description": (
            "Citation and coverage metrics for the local Zotero library: "
            "item counts, PDFs, DOIs, Better BibTeX keys, types, years, largest collections. "
            "Use when the user asks for citation metrics, library stats, or how large the corpus is. "
            "This is local coverage, not live Crossref/Google Scholar citation counts."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

ZOTERO_TOOL_SYSTEM_HINT = (
    "You can read the user's Zotero library with zotero_tree, zotero_search, "
    "zotero_list, zotero_read, and zotero_metrics. When they mention their zotero tree, "
    "papers, citations, PDFs, or a term in the library, call these tools. "
    "Do not use shell_ls or vault_* to find Zotero. Do not claim you cannot access Zotero. "
    "Do not dump the entire library. Search or list a collection, then zotero_read specific keys."
)


def zotero_chat_tools() -> list[dict[str, Any]]:
    if not zotero_tools_enabled():
        return []
    return [
        ZOTERO_TREE_TOOL,
        ZOTERO_SEARCH_TOOL,
        ZOTERO_LIST_TOOL,
        ZOTERO_READ_TOOL,
        ZOTERO_METRICS_TOOL,
    ]


def execute_zotero_tool(name: str, arguments: dict[str, Any]) -> str:
    client = ZoteroClient()
    if name == ZOTERO_TREE:
        collection = str(arguments.get("collection") or "").strip()
        depth = arguments.get("depth", 2)
        try:
            levels = int(depth)
        except (TypeError, ValueError):
            levels = 2
        return format_tool_payload(client.format_tree(collection=collection, depth=levels))
    if name == ZOTERO_SEARCH:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "error: query is required"
        collection = str(arguments.get("collection") or "").strip()
        limit = arguments.get("limit", 20)
        try:
            n = int(limit)
        except (TypeError, ValueError):
            n = 20
        return format_tool_payload(client.search_items(query, collection=collection, limit=n))
    if name == ZOTERO_LIST:
        collection = str(arguments.get("collection") or "").strip()
        if not collection:
            return "error: collection is required"
        found = client.resolve_collection(collection)
        if found is None:
            return format_tool_payload({"error": f"collection not found: {collection}"})
        limit = arguments.get("limit", 40)
        try:
            n = int(limit)
        except (TypeError, ValueError):
            n = 40
        items = client.list_collection_items(found.key, limit=max(1, min(n, 100)))
        return format_tool_payload(
            {
                "collection": found.name,
                "key": found.key,
                "count": len(items),
                "items": [
                    {
                        "key": item.key,
                        "title": item.title,
                        "year": item.year,
                        "creators": item.creators,
                        "item_type": item.item_type,
                    }
                    for item in items
                ],
            }
        )
    if name == ZOTERO_READ:
        key = str(arguments.get("key") or "").strip()
        if not key:
            return "error: key is required"
        include_pdf = bool(arguments.get("include_pdf_text"))
        return client.item_record(key, include_pdf_text=include_pdf)
    if name == ZOTERO_METRICS:
        return format_tool_payload(client.library_metrics())
    return f"error: unknown zotero tool {name!r}"
