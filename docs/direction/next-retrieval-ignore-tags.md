# Plan: retrieval `.ignore` + tagging

Umbrella: [next-mcp-retrieval-loop.md](next-mcp-retrieval-loop.md). Corpus walk today: `src/processing/text/retrieval/corpus/sources.py` (`_SKIP_DIR_NAMES` hardcoded). Build: `sophon-rag-index` / `/rag-index`.

## Goal

Let the human control **what enters the default vault+project corpus** and **how chunks are labeled for retrieve**, without inventing a second index product.

## Locks

1. Hardcoded skip dirs stay as the safety floor (`.git`, `.venv`, `node_modules`, `models`, …). Ignore files only **add** exclusions (and optional includes later). They never re-admit secrets dirs we already skip.
2. One ignore grammar for vault and project roots. Prefer gitignore-style patterns.
3. Tags are metadata on corpus units (file or chunk), not a parallel embedding store in v1.
4. Adaptive-RAG gate stays. Tags feed filters / boosts after retrieve or as query hints. They do not replace the skip / single_hop / multi_hop decision.
5. Rebuild is explicit. Changing ignore or tags does nothing until `/rag-index --rebuild` (or CLI equivalent).
6. Paths in index records still note which root (WSL vs Windows) when relevant ([os-and-deploy.md](os-and-deploy.md)).

## Substeps

### A. `.ignore` (or `.ragignore`)

1. **File locations**
   - Project: `.sophon/rag.ignore` (repo-local, shareable) and optional `.sophon/rag.ignore.local` (gitignored).
   - Vault: `<vault>/.ragignore` next to Obisdian root, or `SOPHON_RAG_IGNORE` path override.
   - Name lock at implement time: one of `.ragignore` / `rag.ignore`. Document the winner. Do not support five aliases.

2. **Walker**
   - Extend `collect_corpus_files` to honor patterns after `_should_skip_dir` / suffix checks.
   - Patterns: directory globs, file globs, negation (`!`) if the chosen library supports it.
   - Log skip counts in `/rag-index` progress and `/rag-status` (files kept / skipped by ignore).

3. **Slash / CLI**
   - `/rag-ignore` prints active files + first N patterns + resolved root.
   - `sophon-rag-index --dry-run` lists would-be files without writing the index (optional same PR).

### B. Tagging

1. **Sources of tags**
   - Vault: Obsidian frontmatter `tags:` / `tag:` on notes.
   - Project: optional YAML/TOML frontmatter or a sidecar `.sophon/rag-tags.yaml` mapping globs → tags.
   - Manual: `/rag-tag PATH tag1,tag2` writes sidecar or frontmatter via editor Review (no silent disk write from the model).

2. **Index payload**
   - Store `tags: list[str]` on each chunk (or file-level inherited by chunks) in LEANN metadata / extras.
   - `/rag-probe` and retrieve `extras` surface tags for debugging.

3. **Query use (v1 minimal)**
   - Optional filter: `/rag-probe --tag sed` or tool param `tags_any=[...]`.
   - Untag: `/rag-untag PATH tag` or edit frontmatter + rebuild.
   - No learned tagger. No auto-tag from embeddings in this plan.

4. **Eval hook**
   - Extend `/eval rag` notes when ignore/tag filters change hit rate (document in eval output, no new test script unless asked).

## Done when

- A vault or project ignore file excludes matching paths from a rebuild.
- At least one tag path (frontmatter **or** sidecar) appears on retrieved chunks.
- `/rag-status` shows ignore file path(s) and whether tags are enabled.
- Default behavior with no ignore/tag files matches today’s corpus.

## Out

- Separate per-tag indexes
- Mem0 / external memory MCP as the tag store
- Replacing Adaptive-RAG with a learned router
- Editor graph / tag browser UI (optional later under editor.md)
