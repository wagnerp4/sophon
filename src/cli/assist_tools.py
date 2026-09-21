from __future__ import annotations

from typing import Any

from cli.code_assist import (
    FileEdit,
    apply_changeset_to_disk,
    apply_old_new,
    current_file_text,
    display_path,
    editor_tools_enabled,
    ensure_assist,
    notify_assist_ui,
    resolve_assist_path,
    truncate_text,
    unified_diff_text,
)

EDITOR_READ = "editor_read"
EDITOR_PROPOSE = "editor_propose_edit"
EDITOR_STATUS = "editor_status"

EDITOR_TOOL_NAMES = frozenset({EDITOR_READ, EDITOR_PROPOSE, EDITOR_STATUS})

EDITOR_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": EDITOR_READ,
        "description": (
            "Read a text file as the editor sees it. Uses the open buffer when that file is open, "
            "then any pending proposal, then disk. Empty path reads the open editor file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or workspace-relative file path. Empty = open editor file.",
                },
            },
            "additionalProperties": False,
        },
    },
}

EDITOR_PROPOSE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": EDITOR_PROPOSE,
        "description": (
            "Propose an edit without writing disk. The user reviews it in the editor Review tab "
            "(Accept all / Decline all). Prefer unique old_string/new_string replacements. "
            "Use content for a full-file replacement or a new file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to edit. Empty = the open editor file.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace. Must match once in the current file.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement for old_string.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file contents after the edit. Use for new files or whole-file rewrites.",
                },
                "description": {
                    "type": "string",
                    "description": "Short summary of the change.",
                },
            },
            "additionalProperties": False,
        },
    },
}

EDITOR_STATUS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": EDITOR_STATUS,
        "description": "List pending (affected) files and accepted (edited) files for this assist session.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

EDITOR_TOOL_SYSTEM_HINT = (
    "You can propose file edits with editor_read, editor_propose_edit, and editor_status. "
    "Proposals stay pending until the user Accepts them in the editor Review tab. "
    "Paths outside the workspace require a harness permission prompt. "
    "Do not use shell_exec to write source files when these tools are available. "
    "Prefer unique old_string/new_string replacements. Use content for new files or full rewrites."
)


def editor_chat_tools() -> list[dict[str, Any]]:
    if not editor_tools_enabled():
        return []
    return [EDITOR_READ_TOOL, EDITOR_PROPOSE_TOOL, EDITOR_STATUS_TOOL]


def execute_editor_tool(state: object, name: str, arguments: dict[str, Any]) -> str:
    if name == EDITOR_READ:
        return _execute_read(state, arguments)
    if name == EDITOR_PROPOSE:
        return _execute_propose(state, arguments)
    if name == EDITOR_STATUS:
        return _execute_status(state)
    return f"error: unknown editor tool {name!r}"


def accept_pending_edits(state: object) -> str:
    controller = ensure_assist(state)
    changeset = controller.accept_pending()
    if changeset is None:
        return "no pending edits"
    errors = apply_changeset_to_disk(changeset, direction="after")
    notify_assist_ui(state)
    root = _workspace(state)
    body = "accepted:\n" + changeset.summary(root)
    if errors:
        return body + "\nwrite errors:\n" + "\n".join(errors)
    return body


def decline_pending_edits(state: object) -> str:
    controller = ensure_assist(state)
    changeset = controller.decline_pending()
    if changeset is None:
        return "no pending edits"
    notify_assist_ui(state)
    return "declined:\n" + changeset.summary(_workspace(state))


def undo_accepted_edits(state: object) -> str:
    controller = ensure_assist(state)
    changeset = controller.undo_accepted()
    if changeset is None:
        return "nothing to undo"
    errors = apply_changeset_to_disk(changeset, direction="before")
    notify_assist_ui(state)
    body = "undid:\n" + changeset.summary(_workspace(state))
    if errors:
        return body + "\nwrite errors:\n" + "\n".join(errors)
    return body


def redo_accepted_edits(state: object) -> str:
    controller = ensure_assist(state)
    changeset = controller.redo_accepted()
    if changeset is None:
        return "nothing to redo"
    errors = apply_changeset_to_disk(changeset, direction="after")
    notify_assist_ui(state)
    body = "redid:\n" + changeset.summary(_workspace(state))
    if errors:
        return body + "\nwrite errors:\n" + "\n".join(errors)
    return body


def _workspace(state: object):
    workspace = getattr(state, "editor_workspace", None)
    if workspace is not None:
        return workspace
    return getattr(state, "project_root", None)


def _execute_read(state: object, arguments: dict[str, Any]) -> str:
    try:
        path = resolve_assist_path(state, str(arguments.get("path") or ""))
        text = current_file_text(state, path)
    except (OSError, ValueError) as exc:
        return f"error: {exc}"
    pending = ensure_assist(state).pending_edit(path)
    origin = "pending proposal" if pending is not None else "editor/disk"
    header = f"{path} ({origin}, {len(text)} chars)"
    return truncate_text(header + "\n" + text)


def _execute_propose(state: object, arguments: dict[str, Any]) -> str:
    try:
        path = resolve_assist_path(state, str(arguments.get("path") or ""))
        current = current_file_text(state, path)
    except (OSError, ValueError) as exc:
        return f"error: {exc}"
    description = str(arguments.get("description") or "").strip()
    if "content" in arguments and arguments.get("content") is not None:
        after = str(arguments.get("content"))
    else:
        old_string = str(arguments.get("old_string") or "")
        new_string = str(arguments.get("new_string") or "")
        if not old_string:
            if current:
                return "error: provide old_string/new_string or content"
            after = new_string
        else:
            try:
                after = apply_old_new(current, old_string, new_string)
            except ValueError as exc:
                return f"error: {exc}"
    if current == after:
        return f"no change: {path}"
    # TODO: apply unified_diff arguments from the model
    edit = FileEdit(path=path, before=current, after=after, description=description)
    stored = ensure_assist(state).add_edit(edit)
    notify_assist_ui(state)
    root = _workspace(state)
    label = display_path(stored.path, root)
    if stored.before == stored.after:
        return f"cleared pending edit for {label}"
    diff = truncate_text(unified_diff_text(stored.path, stored.before, stored.after, root), 4_000)
    return (
        f"pending {stored.kind} {label} {stored.stats}\n"
        f"{diff}\n"
        "Queued for Review. The user must Accept all or Decline all."
    )


def _execute_status(state: object) -> str:
    return ensure_assist(state).status_text(_workspace(state))
