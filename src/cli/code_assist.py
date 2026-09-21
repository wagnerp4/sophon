from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from pathlib import Path

_MAX_READ_CHARS = 12_000
_MAX_EXCERPT_CHARS = 4_000


def editor_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_EDITOR_TOOLS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def display_path(path: Path, root: Path | None) -> str:
    resolved = path
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if root is not None:
        try:
            return str(resolved.relative_to(root.resolve())).replace("\\", "/")
        except (OSError, ValueError):
            pass
    return str(resolved)


def diff_stats(before: str, after: str) -> str:
    old_lines = before.splitlines()
    new_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    added = 0
    removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return f"+{added}/-{removed}"


def unified_diff_text(path: Path, before: str, after: str, root: Path | None = None) -> str:
    label = display_path(path, root)
    old = before.splitlines(keepends=True)
    new = after.splitlines(keepends=True)
    if not old and not before:
        old = []
    if not new and not after:
        new = []
    rendered = "".join(
        difflib.unified_diff(
            old,
            new,
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            lineterm="\n",
        )
    )
    if rendered.strip():
        return rendered
    return f"--- a/{label}\n+++ b/{label}\n(no textual difference)\n"


def truncate_text(text: str, limit: int = _MAX_READ_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def apply_old_new(current: str, old_string: str, new_string: str) -> str:
    if old_string not in current:
        raise ValueError("old_string was not found in the current file")
    matches = current.count(old_string)
    if matches != 1:
        raise ValueError(f"old_string matched {matches} times; include more surrounding context")
    return current.replace(old_string, new_string, 1)


@dataclass
class FileEdit:
    path: Path
    before: str
    after: str
    description: str = ""

    @property
    def kind(self) -> str:
        if self.before == "" and self.after != "":
            return "add"
        if self.before != "" and self.after == "":
            return "delete"
        return "modify"

    @property
    def stats(self) -> str:
        return diff_stats(self.before, self.after)


@dataclass
class ChangeSet:
    edits: dict[str, FileEdit] = field(default_factory=dict)

    def file_list(self) -> list[FileEdit]:
        return sorted(self.edits.values(), key=lambda item: str(item.path).lower())

    def edit_for(self, path: Path) -> FileEdit | None:
        return self.edits.get(_path_key(path))

    def summary(self, root: Path | None = None) -> str:
        if not self.edits:
            return "(empty)"
        lines = [f"{len(self.edits)} file(s)"]
        for edit in self.file_list():
            label = display_path(edit.path, root)
            note = f" {edit.description}" if edit.description else ""
            lines.append(f"  {edit.kind[0].upper()} {label} {edit.stats}{note}")
        return "\n".join(lines)


class CodeAssistController:
    def __init__(self) -> None:
        self.pending: ChangeSet | None = None
        self.undo_stack: list[ChangeSet] = []
        self.redo_stack: list[ChangeSet] = []
        self.last_applied: ChangeSet | None = None
        self.last_direction: str | None = None

    def pending_edit(self, path: Path) -> FileEdit | None:
        if self.pending is None:
            return None
        return self.pending.edits.get(_path_key(path))

    def add_edit(self, edit: FileEdit) -> FileEdit:
        if self.pending is None:
            self.pending = ChangeSet()
        key = _path_key(edit.path)
        existing = self.pending.edits.get(key)
        if existing is not None:
            edit = FileEdit(
                path=existing.path,
                before=existing.before,
                after=edit.after,
                description=edit.description or existing.description,
            )
        if edit.before == edit.after:
            self.pending.edits.pop(key, None)
            if not self.pending.edits:
                self.pending = None
            return edit
        self.pending.edits[key] = edit
        return edit

    def accept_pending(self) -> ChangeSet | None:
        changeset = self.pending
        if changeset is None or not changeset.edits:
            self.pending = None
            return None
        self.pending = None
        self.undo_stack.append(changeset)
        self.redo_stack.clear()
        self.last_applied = changeset
        self.last_direction = "after"
        return changeset

    def decline_pending(self) -> ChangeSet | None:
        changeset = self.pending
        self.pending = None
        self.last_applied = None
        self.last_direction = None
        return changeset

    def undo_accepted(self) -> ChangeSet | None:
        if not self.undo_stack:
            return None
        changeset = self.undo_stack.pop()
        self.redo_stack.append(changeset)
        self.last_applied = changeset
        self.last_direction = "before"
        return changeset

    def redo_accepted(self) -> ChangeSet | None:
        if not self.redo_stack:
            return None
        changeset = self.redo_stack.pop()
        self.undo_stack.append(changeset)
        self.last_applied = changeset
        self.last_direction = "after"
        return changeset

    def edited_files(self) -> list[FileEdit]:
        latest: dict[str, FileEdit] = {}
        for changeset in self.undo_stack:
            for key, edit in changeset.edits.items():
                latest[key] = edit
        return sorted(latest.values(), key=lambda item: str(item.path).lower())

    def status_text(self, root: Path | None = None) -> str:
        lines: list[str] = []
        if self.pending is not None and self.pending.edits:
            lines.append("affected (pending):")
            lines.append(self.pending.summary(root))
        else:
            lines.append("affected (pending): none")
        edited = self.edited_files()
        if edited:
            lines.append("edited (accepted):")
            for edit in edited:
                label = display_path(edit.path, root)
                lines.append(f"  {edit.kind[0].upper()} {label} {edit.stats}")
        else:
            lines.append("edited (accepted): none")
        lines.append(f"undo stack: {len(self.undo_stack)}")
        lines.append(f"redo stack: {len(self.redo_stack)}")
        return "\n".join(lines)


def ensure_assist(state: object) -> CodeAssistController:
    current = getattr(state, "assist", None)
    if isinstance(current, CodeAssistController):
        return current
    controller = CodeAssistController()
    setattr(state, "assist", controller)
    return controller


def resolve_assist_path(state: object, raw: str) -> Path:
    text = (raw or "").strip()
    workspace = getattr(state, "editor_workspace", None)
    root = Path(workspace) if workspace else Path(getattr(state, "project_root", Path.cwd()))
    try:
        root = root.expanduser().resolve()
    except OSError:
        root = Path(root).expanduser()
    if not text:
        open_path = getattr(state, "editor_open_path", None)
        if open_path:
            return Path(str(open_path))
        raise ValueError("path is required when no file is open")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def current_file_text(state: object, path: Path) -> str:
    controller = ensure_assist(state)
    pending = controller.pending_edit(path)
    if pending is not None:
        return pending.after
    open_path = getattr(state, "editor_open_path", None)
    open_text = getattr(state, "editor_open_text", None)
    if open_path and open_text is not None:
        try:
            if Path(str(open_path)).resolve() == path.resolve():
                return str(open_text)
        except OSError:
            if Path(str(open_path)) == path:
                return str(open_text)
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def apply_changeset_to_disk(changeset: ChangeSet, *, direction: str) -> list[str]:
    errors: list[str] = []
    for edit in changeset.file_list():
        text = edit.after if direction == "after" else edit.before
        try:
            if direction == "before" and edit.kind == "add":
                if edit.path.is_file():
                    edit.path.unlink()
                continue
            if direction == "after" and edit.kind == "delete":
                if edit.path.is_file():
                    edit.path.unlink()
                continue
            edit.path.parent.mkdir(parents=True, exist_ok=True)
            edit.path.write_text(text, encoding="utf-8")
        except OSError as exc:
            errors.append(f"{edit.path}: {exc}")
    return errors


def editor_context_hint(state: object) -> str | None:
    path = getattr(state, "editor_open_path", None)
    text = getattr(state, "editor_open_text", None)
    controller = ensure_assist(state)
    lines: list[str] = []
    if path:
        lines.append(f"Open editor file: {path}")
        if isinstance(text, str):
            if not text:
                lines.append("Open buffer is empty.")
            elif len(text) <= _MAX_EXCERPT_CHARS:
                lines.append("Open buffer:\n```\n" + text + "\n```")
            else:
                lines.append(f"Open buffer is {len(text)} chars. Call editor_read before editing.")
    if controller.pending is not None and controller.pending.edits:
        lines.append("Pending proposals:\n" + controller.pending.summary(getattr(state, "editor_workspace", None)))
    if not lines:
        return None
    return "\n".join(lines)


def notify_assist_ui(state: object) -> None:
    hook = getattr(state, "assist_ui_hook", None)
    if callable(hook):
        hook()


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)
