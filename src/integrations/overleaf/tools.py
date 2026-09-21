from __future__ import annotations

from typing import Any

from integrations.overleaf.client import (
    format_tool_payload,
    list_files,
    list_projects,
    list_sections,
    overleaf_tools_enabled,
    read_file,
)

OVERLEAF_LIST_PROJECTS = "overleaf_list_projects"
OVERLEAF_LIST = "overleaf_list"
OVERLEAF_READ = "overleaf_read"
OVERLEAF_SECTIONS = "overleaf_sections"

OVERLEAF_TOOL_NAMES = frozenset(
    {
        OVERLEAF_LIST_PROJECTS,
        OVERLEAF_LIST,
        OVERLEAF_READ,
        OVERLEAF_SECTIONS,
    }
)

OVERLEAF_LIST_PROJECTS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": OVERLEAF_LIST_PROJECTS,
        "description": (
            "List configured Overleaf Git projects (ids, aliases, sync status). "
            "Use before overleaf_list / overleaf_read. Do not use shell_ls for Overleaf."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

OVERLEAF_LIST_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": OVERLEAF_LIST,
        "description": (
            "List files and folders in an Overleaf project directory "
            "(relative to the project root). Empty path lists the root."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Overleaf project id (from overleaf_list_projects).",
                },
                "path": {
                    "type": "string",
                    "description": "Project-relative folder path (default empty = root).",
                },
            },
            "required": ["project_id"],
            "additionalProperties": False,
        },
    },
}

OVERLEAF_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": OVERLEAF_READ,
        "description": (
            "Read a text file from an Overleaf project by project-relative path "
            "(e.g. main.tex or chapters/intro.tex)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Overleaf project id.",
                },
                "path": {
                    "type": "string",
                    "description": "Project-relative file path.",
                },
            },
            "required": ["project_id", "path"],
            "additionalProperties": False,
        },
    },
}

OVERLEAF_SECTIONS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": OVERLEAF_SECTIONS,
        "description": (
            "List LaTeX section / subsection / subsubsection headings in a .tex file "
            "with line numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Overleaf project id.",
                },
                "path": {
                    "type": "string",
                    "description": "Project-relative .tex path.",
                },
            },
            "required": ["project_id", "path"],
            "additionalProperties": False,
        },
    },
}

OVERLEAF_TOOL_SYSTEM_HINT = (
    "You can read Overleaf Git projects with overleaf_list_projects, overleaf_list, "
    "overleaf_read, and overleaf_sections. Prefer overleaf_list_projects first, then "
    "overleaf_list / overleaf_read. Do not invent project ids or file paths. "
    "Writes and compile are not available yet."
)


def overleaf_chat_tools() -> list[dict[str, Any]]:
    if not overleaf_tools_enabled():
        return []
    return [
        OVERLEAF_LIST_PROJECTS_TOOL,
        OVERLEAF_LIST_TOOL,
        OVERLEAF_READ_TOOL,
        OVERLEAF_SECTIONS_TOOL,
    ]


def execute_overleaf_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == OVERLEAF_LIST_PROJECTS:
        rows = []
        for project in list_projects():
            rows.append(
                {
                    "project_id": project.project_id,
                    "alias": project.alias,
                    "path": str(project.path),
                    "sync_error": project.sync_error,
                }
            )
        return format_tool_payload({"projects": rows})
    if name == OVERLEAF_LIST:
        project_id = str(arguments.get("project_id") or "").strip()
        if not project_id:
            return "error: project_id is required"
        path = str(arguments.get("path") or "").strip()
        entries = list_files(project_id, path)
        return format_tool_payload(
            {
                "project_id": project_id,
                "path": path or "/",
                "entries": [
                    {
                        "name": entry.name,
                        "path": entry.relpath,
                        "kind": "dir" if entry.is_dir else "file",
                    }
                    for entry in entries
                ],
            }
        )
    if name == OVERLEAF_READ:
        project_id = str(arguments.get("project_id") or "").strip()
        path = str(arguments.get("path") or "").strip()
        if not project_id:
            return "error: project_id is required"
        if not path:
            return "error: path is required"
        return read_file(project_id, path, truncate=True)
    if name == OVERLEAF_SECTIONS:
        project_id = str(arguments.get("project_id") or "").strip()
        path = str(arguments.get("path") or "").strip()
        if not project_id:
            return "error: project_id is required"
        if not path:
            return "error: path is required"
        return format_tool_payload(
            {
                "project_id": project_id,
                "path": path,
                "sections": list_sections(project_id, path),
            }
        )
    return f"error: unknown overleaf tool {name!r}"
