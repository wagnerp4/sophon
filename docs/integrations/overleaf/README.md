# Overleaf Git projects in sophon

The editor project pane can show Overleaf Premium Git projects (clone/pull into a local cache). Chat tools can list and read files from those clones.

## Prerequisites

1. Overleaf **Premium** with Git Integration enabled.
2. Create a token: **Account Settings → Git Integration → Create Token**.
3. Project id from the browser URL: `https://www.overleaf.com/project/<PROJECT_ID>`.
4. Local `git` on `PATH`.

## Env

Prefer the sophon repo `.env` (loaded via `load_sophon_dotenv`):

```text
SOPHON_OVERLEAF_TOOLS=1
SOPHON_OVERLEAF_GIT_TOKEN=olp_...
SOPHON_OVERLEAF_PROJECT_ID=your_project_id
# or several:
# SOPHON_OVERLEAF_PROJECT_IDS=id1,id2
```

`OVERLEAF_GIT_TOKEN` is also honored. Tools stay off until `SOPHON_OVERLEAF_TOOLS=1` and a token plus at least one project id are set.

Optional aliases file `data/overleaf/projects.json`:

```json
{
  "ssl4sed-paper": "abcdef0123456789abcdef01"
}
```

Clones live under `data/overleaf/projects/<id>/` (gitignored via `data/*`). The token is used for clone/pull only. Origin is rewritten to a tokenless URL after clone.

## Editor tree

In Editor, **View → Overleaf tree** (or cycle with `Ctrl+Shift+T`) replaces the project pane with configured projects. Expanding a project syncs (clone or `git pull --ff-only`) and lists files. Selecting a `.tex` / `.bib` / text file opens it read-only in the source pane (no Save to Overleaf yet). PDFs and images in the clone use the existing preview path.

The terminal cwd stays on the code workspace.

## Chat tools (LM Studio)

When enabled, LM Studio chat gets `overleaf_list_projects`, `overleaf_list`, `overleaf_read`, and `overleaf_sections`. Restart chat after `.env` changes.

Slash commands:

```text
/overleaf-status
/overleaf-list
/overleaf-list PROJECT_ID
/overleaf-list PROJECT_ID chapters
/overleaf-read PROJECT_ID main.tex
```

## Not wired yet

- Write / `update_section` + `git commit` / `git push`
- Unlocking editor Save for Overleaf files
- Browser-session compile / build logs / PDF fetch from Overleaf
- Cursor npm MCP wrappers (`@mjyoo2/overleaf-mcp`, etc.) as a separate Cursor-side option
