# Zotero library in sophon

The editor project pane can show the local Zotero collection tree (same library the Zotero MCP / `zotero-cli` uses).

## Prerequisites

1. Zotero desktop running, with **Settings → Advanced → Allow other applications on this computer to communicate with Zotero** enabled. The local API is `http://127.0.0.1:23119/api`.
2. If Zotero is not running, sophon falls back to a read-only sqlite open of `zotero.sqlite`.

Optional `.env`:

```text
SOPHON_ZOTERO_API_URL=http://127.0.0.1:23119/api
SOPHON_ZOTERO_DB_PATH=C:\Users\Phili\Zotero\zotero.sqlite
```

`ZOTERO_DB_PATH` is also honored (same variable `zotero-mcp` uses).

## Editor tree

In Editor, **View → Zotero tree** (or cycle with `Ctrl+Shift+T`) replaces the project pane with collections. Expanding a collection lists items. Selecting an item opens its PDF in the existing PDF preview when a local file exists. Otherwise the item metadata is shown as locked markdown.

The terminal cwd stays on the code workspace.

## Chat tools (LM Studio)

When the library is reachable, LM Studio chat gets `zotero_tree`, `zotero_search`, `zotero_list`, `zotero_read`, and `zotero_metrics`. Set `SOPHON_ZOTERO_TOOLS=0` to disable. Restart chat after code or `.env` changes.

Slash commands (work without waiting for the model):

```text
/zotero-status
/zotero-tree
/zotero-tree Computer Science 3
/zotero-search retrieval augmented
/zotero-metrics
/zotero-read Mamba
/zotero-read C4VGLK5E --pdf
```

- `zotero_search` matches titles, authors, Better BibTeX keys, and Zotero's indexed PDF words.
- `zotero_metrics` is local coverage (item counts, DOIs, citekeys, years). It is not live Crossref or Google Scholar citation counts.
- `zotero_read` with `include_pdf_text` / `--pdf` appends a truncated local PDF excerpt. Do not ask the model to ingest the entire library.

## Notes

- Group libraries, saved searches, and semantic search are not wired yet.
- Cursor's Zotero MCP (`user-zotero`) and `zotero-cli` remain available for writes.
