from __future__ import annotations

import os
from typing import Any

from .catalog import SkillCatalog

SKILL_LIST = "skill_list"
SKILL_READ = "skill_read"
SKILL_READ_FILE = "skill_read_file"

SKILL_TOOL_NAMES = frozenset({SKILL_LIST, SKILL_READ, SKILL_READ_FILE})

_MAX_FILE_CHARS = 12_000


def skill_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_SKILL_TOOLS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


SKILL_LIST_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SKILL_LIST,
        "description": (
            "List available skills (name plus description). Skills are reusable "
            "procedures. Call this before a specialized workflow to see if a skill "
            "already covers it."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

SKILL_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SKILL_READ,
        "description": (
            "Read the full SKILL.md body of one skill by name. Read before "
            "following a skill; do not invent a skill that is not in skill_list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name from skill_list."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

SKILL_READ_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SKILL_READ_FILE,
        "description": (
            "Read a bundled resource of a skill (references/ or scripts/ file). "
            "Path is relative to the skill directory. Scripts are shown, not run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name from skill_list."},
                "path": {
                    "type": "string",
                    "description": "Relative path under the skill directory, e.g. references/REFERENCE.md.",
                },
            },
            "required": ["name", "path"],
            "additionalProperties": False,
        },
    },
}

SKILL_TOOL_SYSTEM_HINT = (
    "Skills are reusable procedures stored as SKILL.md files. Call skill_list before a "
    "specialized workflow, then skill_read to load one. Do not claim to follow a skill you "
    "did not read, and do not invent skills that are not listed. You cannot write skills; "
    "the user creates them through Review."
)


def skill_chat_tools() -> list[dict[str, Any]]:
    if not skill_tools_enabled():
        return []
    return [SKILL_LIST_TOOL, SKILL_READ_TOOL, SKILL_READ_FILE_TOOL]


def _resolve_within(base, target):
    from pathlib import Path

    root = Path(base).resolve()
    candidate = (root / str(target)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def execute_skill_tool(catalog: SkillCatalog | None, name: str, arguments: dict[str, Any]) -> str:
    if catalog is None:
        return "error: skills are disabled (SOPHON_SKILLS=0)"
    if name == SKILL_LIST:
        if not catalog.entries:
            return "(no skills found)"
        lines = []
        for entry in catalog.entries:
            desc = " ".join(entry.description.split())
            lines.append(f"- {entry.name} (root={entry.root_label}): {desc}")
        return "\n".join(lines)
    if name == SKILL_READ:
        entry = catalog.get(str(arguments.get("name") or ""))
        if entry is None:
            return "not found"
        body = entry.spec.body.strip()
        header = f"# {entry.name} ({entry.root_label})\n{entry.description}\n"
        text = f"{header}\n{body}" if body else header
        if len(text) > _MAX_FILE_CHARS:
            text = text[: _MAX_FILE_CHARS - 1] + "\u2026"
        return text
    if name == SKILL_READ_FILE:
        entry = catalog.get(str(arguments.get("name") or ""))
        if entry is None:
            return "not found"
        rel = str(arguments.get("path") or "").strip()
        if not rel:
            return "error: path is required"
        target = _resolve_within(entry.skill_dir, rel)
        if target is None:
            return "error: path escapes the skill directory"
        if not target.is_file():
            return "not found"
        try:
            data = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"error: {exc}"
        if len(data) > _MAX_FILE_CHARS:
            data = data[: _MAX_FILE_CHARS - 1] + "\u2026"
        return data
    return f"error: unknown skill tool {name!r}"
