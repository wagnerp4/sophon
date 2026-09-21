---
name: create-skill
description: Draft a new Sophon skill in the public Agent Skills SKILL.md format. Use when the user asks to create a skill, add a skill, or write a SKILL.md, or when a repeatable procedure should be captured for reuse.
license: MIT
metadata:
  author: sophon
  version: "1.0"
---

# Create a skill

Skills are reusable procedures stored as a directory with a `SKILL.md` file (YAML frontmatter plus a markdown body) and optional `scripts/`, `references/`, `assets/`. Sophon uses the public Agent Skills format, so a skill written here also loads in Claude and Cursor.

## When to create one

Create a skill for a repeatable method: a class of task the agent will do again. Do not create one for a one-off fact or a bugfix. Facts and decisions go to memory (`/memory note`, `/memory promote`). Repeatable procedure goes to a skill.

## Steps

1. Gather purpose and trigger. Ask what task the skill covers and when it should apply.
2. Choose scope. Project (`.sophon/skills`, commits with the repo) or user (`~/.sophon/skills`, all projects). Default project.
3. Pick a name. Lowercase letters, numbers, single hyphens. Max 64 characters. It must match the directory.
4. Write the description. One or two sentences with both what it does and when to use it. Include trigger words the agent would match. This field drives routing, so be specific.
5. Draft a short body. Instructions, then examples. Keep the `SKILL.md` under about 500 lines. Move long reference material into `references/` and load it only when needed.
6. Queue it. Run `/skill create <name>` to scaffold the folder, then edit the body. Accept the change in Review.

## Description quality

Write in third person. State what and when.

- Good: "Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDFs or when the user mentions PDFs, forms, or document extraction."
- Poor: "Helps with PDFs."

## Constraints

- The model cannot write skills directly. `/skill create` and `/skill import` queue a file edit in Review; the user Accepts.
- Deterministic steps belong in `scripts/`, not in prose the model can skip. Scripts are shown, not run automatically.
- Do not put secrets or tokens in a skill.
