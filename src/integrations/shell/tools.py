from __future__ import annotations

from pathlib import Path
from typing import Any

from integrations.shell.runner import ShellSession, shell_tools_enabled, truncate_output

SHELL_PWD = "shell_pwd"
SHELL_CD = "shell_cd"
SHELL_LS = "shell_ls"
SHELL_READ = "shell_read"
SHELL_EXEC = "shell_exec"

SHELL_TOOL_NAMES = frozenset({SHELL_PWD, SHELL_CD, SHELL_LS, SHELL_READ, SHELL_EXEC})

SHELL_PWD_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SHELL_PWD,
        "description": "Return the current working directory for shell/workspace tools.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

SHELL_CD_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SHELL_CD,
        "description": (
            "Change the working directory used by shell tools. "
            "Relative paths are resolved against the current cwd. "
            "Empty path goes to the user home directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative directory path.",
                },
            },
            "additionalProperties": False,
        },
    },
}

SHELL_LS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SHELL_LS,
        "description": (
            "List files and directories in a path (default: current working directory). "
            "Prefer this over shell_exec for directory listing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to cwd or absolute.",
                },
            },
            "additionalProperties": False,
        },
    },
}

SHELL_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SHELL_READ,
        "description": (
            "Read a text file from disk (truncated). Prefer this over shell_exec for reading files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to cwd or absolute.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

SHELL_EXEC_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SHELL_EXEC,
        "description": (
            "Run a shell command in the current working directory. "
            "On Windows use PowerShell syntax. On Unix use the user shell. "
            "Use shell_cd for directory changes instead of cd when possible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

SHELL_TOOL_SYSTEM_HINT = (
    "You can navigate the local filesystem and run shell commands with shell_pwd, "
    "shell_cd, shell_ls, shell_read, and shell_exec. You are running inside the user's "
    "local orodruin chat session with real disk access. "
    "Do not claim you are a remote cloud service without filesystem access. "
    "Prefer shell_ls/shell_read over shell_exec for browsing files. "
    "On Windows, use PowerShell command syntax."
)


def shell_chat_tools() -> list[dict[str, Any]]:
    if not shell_tools_enabled():
        return []
    return [SHELL_PWD_TOOL, SHELL_CD_TOOL, SHELL_LS_TOOL, SHELL_READ_TOOL, SHELL_EXEC_TOOL]


def execute_shell_tool(session: ShellSession, name: str, arguments: dict[str, Any]) -> str:
    if name == SHELL_PWD:
        return str(session.cwd.resolve())
    if name == SHELL_CD:
        path = arguments.get("path")
        path_s = None if path is None else str(path)
        ok, msg = session.change_directory(path_s)
        return msg if ok else f"error: {msg}"
    if name == SHELL_LS:
        path = str(arguments.get("path") or "").strip()
        target = session.resolve_path(path or ".")
        if not target.exists():
            return f"error: path not found: {target}"
        if not target.is_dir():
            return f"error: not a directory: {target}"
        entries: list[str] = []
        try:
            for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                suffix = "/" if child.is_dir() else ""
                entries.append(f"{child.name}{suffix}")
        except OSError as exc:
            return f"error: {exc}"
        return truncate_output("\n".join(entries) if entries else "(empty)")
    if name == SHELL_READ:
        path = str(arguments.get("path") or "").strip()
        if not path:
            return "error: path is required"
        target = session.resolve_path(path)
        if not target.is_file():
            return f"error: not a file: {target}"
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"error: {exc}"
        return truncate_output(text)
    if name == SHELL_EXEC:
        command = str(arguments.get("command") or "").strip()
        if not command:
            return "error: command is required"
        result = session.run(command)
        body = result.output.strip() or f"(exit {result.exit_code})"
        if result.handled_as_cd:
            return body
        if result.exit_code != 0 and f"(exit {result.exit_code})" not in body:
            return f"{body}\n(exit {result.exit_code})"
        return body
    return f"error: unknown shell tool {name!r}"


def ensure_shell_session(cwd: Path | None = None) -> ShellSession:
    root = cwd if cwd is not None else Path.cwd()
    return ShellSession(cwd=Path(root).resolve())
