# Editor

## Role

The editor is the filesystem truth for the human and the landing zone for harness-proposed diffs. It is not a VS Code clone. It is the tree + buffer + preview + review surface that a terminal engineer can live in for the files this project actually touches.

## Current behavior

- Mode: `Ctrl+E`. Trees: Project, Obsidian vault (`SOPHON_VAULT_PATH`), Zotero, Drive, Bookmarks, Overleaf (read), disk roots (`C:\`, `H:\`, …) via View. Cycle: `Ctrl+Shift+T` or View menu. Cwd of the process does not change when the tree changes.
- Save / refresh: `Ctrl+S`, `Ctrl+R`. Keybinds: `.sophon/keybinds.yaml` (reload on save from the editor).
- Previews: Markdown, SVG, STL (orbit), tables (xlsx preview-only), ipynb (stored cells, no kernel), PDF (page text default, optional current-page raster), raster images.
- Python buffers are full-width with syntax highlight when `textual[syntax]` is installed.
- Lower Terminal tab: one shell command at a time (PowerShell on Windows).
- Code review tab: combined changeset from editor tools. Accept all / Decline all / Undo batch / Redo batch.

Done vs open is listed in [TODO.md](../TODO.md) (Overleaf write, tags/graph/recent, `.tex` to PDF, language pack, smart completion).

## Keybinds

- **Profile:** `.sophon/keybinds.yaml` in the sophon install root. It applies everywhere. `/keybind` lists it, and also lists entries it rejected. `/keybind set ACTION KEY` edits one entry and keeps the rest of the file.
- **Repo file:** `.sophon/keybinds.yaml` in the folder the editor shows. It only overlays the profile, and only for editor-local actions: `save_file`, `save_as`, `new_file`, `new_folder`, `open_file`, `open_folder`, `find_in_file`, `toggle_project`, `toggle_aux`, `accept_completion`. Repo files cannot bind slash commands, navigation, permission keys, `listen`, or any other action.
- **Key rules:**
  - Every value is exactly one key: up to three modifiers plus a key name, for example `ctrl+s`, `ctrl+shift+o`, `f5`, `right`. No commas or spaces.
  - Repo keys also need a modifier or an F-key, unless the key is the profile's own key for that action.
  - Repo keys must not reuse a key the profile already binds, or a reserved key (`ctrl+q`, `ctrl+c`, `ctrl+p`, `tab`, `shift+tab`, `f1`, `super+c`).
- **Profile slash bindings** may carry arguments (`"tts on": f5`), but not quotes, parentheses, backslashes, or control characters.
- **Safe loading:** files over 64 KiB, YAML aliases, and invalid YAML are ignored as a whole. Loading never writes into a repo.

## Constraints

- **Propose then accept.** Models do not `open().write` on the project tree. Same path for swarm CodeAgents.
- **Preview is not execution.** Notebooks do not start kernels. STL/PDF raster must not block the UI (Windows: no CSI autodetect).
- **Trees are views, not mounts.** Switching to vault or Drive does not `chdir`. Shell tab stays on project root unless the user types a `cd`.
- **Size limits.** `editor_read` truncates (`code_assist` caps). Huge files are grep/search, not full buffer.
- **Binary and generated dirs.** `.venv`, `models/`, `node_modules` stay out of default tree walk where already ignored.
- **Cross-OS paths.** Display with `/` in tool output. Windows deploy tree and WSL checkout are different roots. Editor on Windows sees `C:\Software\...`. Do not write WSL paths into Windows files.

## Future direction

1. Language modes: Python is in. Add Textual syntax for Rust, C family, JS/TS, Java, Go, shell. Missing grammars degrade to plain text, not a crash.
2. Completion: keep 2-char prefix. "Smart" completion means project-local structure (imports, symbols) plus optional harness-backed suggest. Do not vendor a full LSP stack until a single language server is specified.
3. Structural replace: rules for large edits (AST / format-preserving) so agents do not dump whole files. Document rules under a future `structure/` tree (backlog).
4. Overleaf: write, `update_section`, compile, git commit/push. Read-only until auth and conflict UX exist.
5. Vault: tags, graph, recent notes as extra trees, not a second editor product.
6. Drag-and-drop and vision inputs land in Chat, but the editor must show dropped paths as buffers when they are files.
7. `.tex` to PDF rendering as preview, reuse PDF raster path.

## Non-goals

- Debugger, Git GUI, merge tool as v1 editor scope (CLI git via shell is enough until specified).
- Multi-root workspaces that hide which tree an edit belongs to.
- Reintroducing a Workshop pane. Agent control lives in Chat + Review + future swarm overlay.
