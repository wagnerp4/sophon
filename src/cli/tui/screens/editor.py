from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Markdown,
    RichLog,
    Static,
    TextArea,
)

from backend.chat_resolve import is_server_backend
from cli.chat import chat_session_ready, dispatch_chat_line, run_chat_generation
from cli.chat_display import (
    format_assistant_body_line,
    format_assistant_turn_header,
    format_system_line,
    format_user_turn_line,
    model_identity_line,
)

from cli.tui.keybinds import (
    DEFAULT_KEYBINDS,
    ensure_keybinds_file,
    format_key_label,
    keybinds_path,
    load_keybinds,
)
from cli.tui.menubar import MenuAction, MenuGroup, MenuItem, PaneMenuBar
from cli.tui.notebook import NotebookPreview
from cli.tui.path_picker import NamePromptModal, PathPickerModal
from cli.tui.preview import Mesh, RasterPreview, parse_stl_bytes, parse_stl_text
from cli.tui.screens.nav import NAV_BINDINGS, ModeNavigationMixin, compose_nav_bar
from cli.tui.screens.workshop import ProjectTree
from cli.tui.source import EditorSourceTextArea
from cli.tui.split import PaneSplitter
from cli.tui.tables import TableData, load_table
from processing.text.completion import PrefixCompleter, load_completer
from utils.device.env_bootstrap import orodruin_project_root

_AUX_TABS = (
    ("editor-aux-tab-terminal", "editor-aux-terminal"),
    ("editor-aux-tab-chat", "editor-aux-chat"),
    ("editor-aux-tab-logs", "editor-aux-logs"),
    ("editor-aux-tab-errors", "editor-aux-errors"),
)
_AUX_DEFAULT_HEIGHT = 10
_FILES_DEFAULT_WIDTH = 48

_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_PYTHON_SUFFIXES = {".py", ".pyi"}
_SVG_SUFFIXES = {".svg"}
_MESH_SUFFIXES = {".stl"}
_TABLE_SUFFIXES = {".csv", ".tsv", ".tab", ".jsonl", ".xlsx"}
_NOTEBOOK_SUFFIXES = {".ipynb"}
_PDF_SUFFIXES = {".pdf"}
_CONFIG_SUFFIXES = {".toml", ".yaml", ".yml"}
_EDITABLE_SUFFIXES = (
    _MARKDOWN_SUFFIXES
    | _PYTHON_SUFFIXES
    | _SVG_SUFFIXES
    | _MESH_SUFFIXES
    | _TABLE_SUFFIXES
    | _NOTEBOOK_SUFFIXES
    | _PDF_SUFFIXES
    | _CONFIG_SUFFIXES
)
# TODO: extend editable suffixes (e.g. .json, .css, .js, .obj, .parquet)
# TODO: optional LSP / diagnostics strip for Python
# TODO: notebook kernel execution
# TODO: xlsx sheet tabs
# TODO: PDF form fill / annotation edit
_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "orodruin.egg-info",
    ".tox",
}
_PREVIEW_DEBOUNCE_S = 0.35


def _terminal_prompt_placeholder() -> str:
    if sys.platform == "win32":
        return "PowerShell command (Enter to run)"
    return "Shell command (Enter to run)"


def _is_markdown(path: Path) -> bool:
    return path.suffix.lower() in _MARKDOWN_SUFFIXES


def _is_python(path: Path) -> bool:
    return path.suffix.lower() in _PYTHON_SUFFIXES


def _is_svg(path: Path) -> bool:
    return path.suffix.lower() in _SVG_SUFFIXES


def _is_mesh(path: Path) -> bool:
    return path.suffix.lower() in _MESH_SUFFIXES


def _is_table(path: Path) -> bool:
    return path.suffix.lower() in _TABLE_SUFFIXES


def _is_notebook(path: Path) -> bool:
    return path.suffix.lower() in _NOTEBOOK_SUFFIXES


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() in _PDF_SUFFIXES


def _is_binary_table(path: Path) -> bool:
    return path.suffix.lower() == ".xlsx"


def _is_editable(path: Path) -> bool:
    return path.suffix.lower() in _EDITABLE_SUFFIXES


def _language_for(path: Path) -> str | None:
    if _is_python(path):
        return "python"
    if _is_markdown(path):
        return "markdown"
    if _is_svg(path):
        return "xml"
    if _is_notebook(path) or path.suffix.lower() == ".jsonl":
        return "json"
    if path.suffix.lower() in {".yaml", ".yml"}:
        return "yaml"
    if path.suffix.lower() == ".toml":
        return "toml"
    return None


class EditorProjectTree(ProjectTree):
    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        kept: list[Path] = []
        for path in paths:
            name = path.name
            suffix = path.suffix.lower()
            if suffix in _EDITABLE_SUFFIXES:
                kept.append(path)
                continue
            if not self._safe_is_dir(path):
                continue
            if name in _SKIP_DIR_NAMES or name.startswith("."):
                continue
            kept.append(path)
        return kept


class UnsavedSwitchModal(ModalScreen[str | None]):
    def compose(self) -> ComposeResult:
        with Vertical(id="editor-unsaved-dialog"):
            yield Static("Unsaved edits in the current file.", id="editor-unsaved-title")
            yield Static("Save before switching, discard changes, or cancel.")
            with Horizontal(id="editor-unsaved-actions"):
                yield Button("Save", id="editor-unsaved-save", variant="primary")
                yield Button("Discard", id="editor-unsaved-discard", variant="warning")
                yield Button("Cancel", id="editor-unsaved-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "editor-unsaved-save": "save",
            "editor-unsaved-discard": "discard",
            "editor-unsaved-cancel": "cancel",
        }
        choice = mapping.get(event.button.id or "")
        self.dismiss(choice)


class EditorScreen(ModeNavigationMixin, Screen):
    BINDINGS = [
        *NAV_BINDINGS,
        Binding("ctrl+c", "absorb_ctrl_c", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._root = orodruin_project_root()
        self._current_path: Path | None = None
        self._saved_text = ""
        self._ignore_changes = False
        self._preview_timer = None
        self._pending_switch: Path | None = None
        self._source_locked = False
        self._mesh: Mesh | None = None
        self._table: TableData | None = None
        self._pdf_bytes: bytes | None = None
        self._preview_mode: str | None = None
        self._preview_user_collapsed = False
        self._files_collapsed = False
        self._files_width = _FILES_DEFAULT_WIDTH
        self._aux_collapsed = False
        self._aux_height = _AUX_DEFAULT_HEIGHT
        self._aux_tab = "editor-aux-terminal"
        self._terminal_busy = False
        self._chat_busy = False
        self._chat_turn = 0
        self._chat_history_synced = False
        self._recent_paths: list[Path] = []
        self._pending_close = False
        self._keybinds: dict[str, str] = dict(DEFAULT_KEYBINDS)
        # TODO: persist recent files/folders across sessions
        self._completer = cast(PrefixCompleter, load_completer("prefix"))

    @property
    def _dirty(self) -> bool:
        if self._current_path is None or self._source_locked:
            return False
        source = self.query_one("#editor-source", EditorSourceTextArea)
        return source.text != self._saved_text

    def compose(self) -> ComposeResult:
        app = self.app
        root = Path(getattr(app, "cwd", self._root)).resolve()
        if not root.is_dir():
            root = self._root
        self._root = root
        self._load_keybinds()
        yield Header(show_clock=True)
        yield from compose_nav_bar("editor")
        with Vertical(id="editor-body"):
            with Horizontal(id="editor-row", classes="editor-flex"):
                yield Button("▸", id="editor-files-show", variant="default")
                with Vertical(id="editor-files-col"):
                    yield PaneMenuBar(
                        "project",
                        f"Project · {root.name}",
                        self._project_menus,
                        id="editor-project-menubar",
                    )
                    yield EditorProjectTree(str(root), id="editor-tree")
                yield PaneSplitter(
                    target_id="editor-files-col",
                    axis="x",
                    min_size=16,
                    flex_min=24,
                    id="editor-files-split",
                )
                with Vertical(id="editor-source-col", classes="editor-flex"):
                    yield PaneMenuBar(
                        "editor",
                        "Editor",
                        self._editor_menus,
                        id="editor-source-menubar",
                    )
                    yield EditorSourceTextArea(
                        "",
                        id="editor-source",
                        soft_wrap=False,
                        show_line_numbers=True,
                        tab_behavior="indent",
                    )
                yield PaneSplitter(
                    target_id="editor-preview-col",
                    axis="x",
                    invert=True,
                    min_size=20,
                    flex_min=24,
                    id="editor-preview-split",
                )
                with Vertical(id="editor-preview-col"):
                    yield PaneMenuBar(
                        "preview",
                        "Preview",
                        self._preview_menus,
                        id="editor-preview-menubar",
                    )
                    with VerticalScroll(id="editor-preview-scroll"):
                        yield Markdown("", id="editor-preview")
                    yield RasterPreview(id="editor-raster")
                    yield DataTable(id="editor-table", zebra_stripes=True, cursor_type="cell")
                    yield NotebookPreview(id="editor-notebook")
            yield PaneSplitter(
                target_id="editor-aux",
                axis="y",
                invert=True,
                min_size=6,
                flex_min=8,
                id="editor-aux-split",
            )
            with Vertical(id="editor-aux"):
                with Horizontal(id="editor-aux-bar"):
                    yield Button("Terminal", id="editor-aux-tab-terminal", variant="primary")
                    yield Button("Chat", id="editor-aux-tab-chat", variant="default")
                    yield Button("Logs", id="editor-aux-tab-logs", variant="default")
                    yield Button("Errors", id="editor-aux-tab-errors", variant="default")
                    yield Static("", id="editor-aux-spacer")
                    yield Button("Hide", id="editor-aux-toggle", variant="default")
                with ContentSwitcher(id="editor-aux-switch", initial="editor-aux-terminal"):
                    with Vertical(id="editor-aux-terminal"):
                        yield Static("", id="editor-aux-terminal-cwd")
                        yield RichLog(
                            id="editor-aux-terminal-log",
                            highlight=False,
                            markup=False,
                            wrap=True,
                            auto_scroll=True,
                        )
                        yield Input(
                            placeholder=_terminal_prompt_placeholder(),
                            id="editor-aux-terminal-prompt",
                        )
                    with Vertical(id="editor-aux-chat"):
                        yield Static("Chat · loading model...", id="editor-aux-chat-status")
                        yield RichLog(
                            id="editor-aux-chat-log",
                            highlight=True,
                            markup=True,
                            wrap=True,
                            auto_scroll=True,
                        )
                        yield Input(
                            placeholder="Message or /command (Enter to send)",
                            id="editor-aux-chat-prompt",
                            disabled=True,
                        )
                    # TODO: attach TUI / model / download logs here
                    yield RichLog(id="editor-aux-logs", highlight=False, markup=True)
                    # TODO: collect save/parse/traceback output here
                    yield RichLog(id="editor-aux-errors", highlight=False, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._apply_keybinds()
        self._set_pane_title("project", f"Project · {self._root.name}")
        self.query_one("#editor-files-show", Button).display = False
        self._set_preview_visible(False)
        self.query_one("#editor-raster", RasterPreview).display = False
        table = self.query_one("#editor-table", DataTable)
        table.add_columns("(empty)")
        table.display = False
        self.query_one("#editor-notebook", NotebookPreview).display = False
        self._refresh_terminal_cwd()
        if sys.platform == "win32":
            self.append_terminal_log("PowerShell · one command per Enter. cls clears this log.")
        else:
            self.append_terminal_log("Shell · one command per Enter. clear clears this log.")
        self.append_terminal_log("This is not an interactive TTY. Environment variables do not persist across commands.")
        self.query_one("#editor-aux-logs", RichLog).write("[dim]Logs will appear here.[/dim]")
        self.query_one("#editor-aux-errors", RichLog).write("[dim]Error traces will appear here.[/dim]")
        source = self.query_one("#editor-source", EditorSourceTextArea)
        source.bind_completer(self._completer)
        self._reindex_project()
        self.refresh_chat_pane()
        self.query_one("#editor-tree", EditorProjectTree).focus()
        self.notify(
            "Select a .py, .md, .svg, .stl, table, .ipynb, .pdf, or .yaml file. Pane File menus open folders. Keybinds: .orodruin/keybinds.yaml.",
            timeout=4,
        )

    def _reindex_project(self) -> None:
        self._completer.index_project(self._root)
        try:
            source = self.query_one("#editor-source", EditorSourceTextArea)
        except NoMatches:
            return
        self._completer.set_buffer_text(source.text)
        source.update_suggestion()

    def _profile_root(self) -> Path:
        return orodruin_project_root()

    def _load_keybinds(self) -> None:
        workspace = self._root
        app_root = self._profile_root()
        if (workspace / ".orodruin" / "keybinds.yaml").is_file():
            self._keybinds = load_keybinds(workspace)
            return
        self._keybinds = load_keybinds(app_root)

    def _apply_keybinds(self) -> None:
        self._load_keybinds()
        binder = getattr(self, "bind", None)
        if binder is None:
            return
        for action, key in self._keybinds.items():
            if not key:
                continue
            try:
                binder(key, action, show=False)
            except Exception:
                self.notify(f"Invalid keybind {action}: {key}")

    def _hotkey(self, action: str) -> str:
        return format_key_label(self._keybinds.get(action, DEFAULT_KEYBINDS.get(action, "")))

    def _set_pane_title(self, pane: str, text: str) -> None:
        mapping = {
            "project": "#editor-project-menubar",
            "editor": "#editor-source-menubar",
            "preview": "#editor-preview-menubar",
        }
        selector = mapping.get(pane)
        if selector is None:
            return
        try:
            self.query_one(selector, PaneMenuBar).set_title(text)
        except NoMatches:
            pass

    def _project_menus(self) -> tuple[MenuGroup, ...]:
        has_file = self._current_path is not None
        can_save = has_file and not self._source_locked
        recent = self._recent_menu_items()
        file_items = (
            MenuItem("New File", "project.new_file", self._hotkey("new_file")),
            MenuItem("New Folder", "project.new_folder", self._hotkey("new_folder")),
            MenuItem("Open File...", "project.open_file", self._hotkey("open_file")),
            MenuItem("Open Folder...", "project.open_folder", self._hotkey("open_folder")),
            *recent,
            MenuItem("Save", "project.save", self._hotkey("save_file"), disabled=not can_save, separator_before=True),
            MenuItem("Save As...", "project.save_as", self._hotkey("save_as"), disabled=self._source_locked),
            MenuItem("Close File", "project.close_file", disabled=not has_file),
            MenuItem("Edit Keybinds...", "project.edit_keybinds", separator_before=True),
        )
        view_items = (
            MenuItem("Refresh Explorer", "project.refresh", self._hotkey("refresh_tree")),
            MenuItem("Collapse Folders", "project.collapse_all"),
            MenuItem(
                "Show Project" if self._files_collapsed else "Hide Project",
                "project.toggle",
                self._hotkey("toggle_project"),
            ),
            MenuItem("Reload Keybinds", "project.reload_keybinds"),
        )
        # TODO: Reveal in OS file manager, New Window, Save All
        return (
            MenuGroup("file", "File", file_items),
            MenuGroup("view", "View", view_items),
        )

    def _editor_menus(self) -> tuple[MenuGroup, ...]:
        has_file = self._current_path is not None
        can_save = has_file and not self._source_locked
        preview_label = "Show Preview" if self._preview_user_collapsed else "Hide Preview"
        project_label = "Show Project" if self._files_collapsed else "Hide Project"
        file_items = (
            MenuItem("Save", "editor.save", self._hotkey("save_file"), disabled=not can_save),
            MenuItem("Save As...", "editor.save_as", self._hotkey("save_as"), disabled=self._source_locked),
            MenuItem("Revert File", "editor.revert", disabled=not has_file),
            MenuItem("Close File", "editor.close_file", disabled=not has_file),
        )
        edit_items = (
            MenuItem("Undo", "editor.undo"),
            MenuItem("Redo", "editor.redo"),
            MenuItem("Cut", "editor.cut", separator_before=True),
            MenuItem("Copy", "editor.copy"),
            MenuItem("Paste", "editor.paste"),
            MenuItem("Find...", "editor.find", self._hotkey("find_in_file"), separator_before=True),
            MenuItem("Select All", "editor.select_all"),
        )
        view_items = (
            MenuItem(preview_label, "editor.toggle_preview", disabled=self._preview_mode is None),
            MenuItem("Toggle Line Numbers", "editor.toggle_line_numbers"),
            MenuItem("Toggle Word Wrap", "editor.toggle_wrap"),
            MenuItem(project_label, "editor.toggle_project", self._hotkey("toggle_project")),
        )
        # TODO: Replace, Go to Line, Format Document, editorconfig / tab-width
        return (
            MenuGroup("file", "File", file_items),
            MenuGroup("edit", "Edit", edit_items),
            MenuGroup("view", "View", view_items),
        )

    def _preview_menus(self) -> tuple[MenuGroup, ...]:
        path = self._current_path
        mesh = path is not None and _is_mesh(path)
        pdf = path is not None and _is_pdf(path)
        view_items = (
            MenuItem("Hide Preview", "preview.hide"),
            MenuItem("Refresh Preview", "preview.refresh"),
            MenuItem("Reset Camera", "preview.reset_camera", disabled=not mesh),
            MenuItem("Previous Page", "preview.page_prev", disabled=not pdf, separator_before=True),
            MenuItem("Next Page", "preview.page_next", disabled=not pdf),
        )
        # TODO: preview zoom, two-page PDF spread, table sheet picker
        return (MenuGroup("view", "View", view_items),)

    def _recent_menu_items(self) -> tuple[MenuItem, ...]:
        items: list[MenuItem] = []
        for index, path in enumerate(self._recent_paths):
            try:
                label = "~/" + path.relative_to(Path.home()).as_posix()
            except ValueError:
                label = str(path)
            if path.is_dir() and not label.endswith("/"):
                label = f"{label}/"
            action = "project.open_recent" if path.is_dir() else "project.open_file_path"
            items.append(
                MenuItem(
                    label,
                    action,
                    payload=str(path),
                    separator_before=index == 0,
                )
            )
        return tuple(items)

    def on_menu_action(self, event: MenuAction) -> None:
        action = event.action
        payload = event.payload
        if action == "project.noop":
            return
        if action == "project.open_recent":
            self._set_workspace(Path(payload))
            return
        if action == "project.open_file_path":
            self._request_open_path(Path(payload))
            return
        handlers = {
            "project.new_file": self.action_new_file,
            "project.new_folder": self.action_new_folder,
            "project.open_file": self.action_open_file,
            "project.open_folder": self.action_open_folder,
            "project.save": self.action_save_file,
            "project.save_as": self.action_save_as,
            "project.close_file": self.action_close_file,
            "project.edit_keybinds": self.action_edit_keybinds,
            "project.refresh": self.action_refresh_tree,
            "project.collapse_all": self.action_collapse_all,
            "project.toggle": self.action_toggle_project,
            "project.reload_keybinds": self.action_reload_keybinds,
            "editor.save": self.action_save_file,
            "editor.save_as": self.action_save_as,
            "editor.revert": self.action_revert_file,
            "editor.close_file": self.action_close_file,
            "editor.undo": lambda: self._run_source_action("action_undo"),
            "editor.redo": lambda: self._run_source_action("action_redo"),
            "editor.cut": lambda: self._run_source_action("action_cut"),
            "editor.copy": lambda: self._run_source_action("action_copy"),
            "editor.paste": lambda: self._run_source_action("action_paste"),
            "editor.find": self.action_find_in_file,
            "editor.select_all": lambda: self._run_source_action("action_select_all"),
            "editor.toggle_preview": self.action_toggle_preview,
            "editor.toggle_line_numbers": self.action_toggle_line_numbers,
            "editor.toggle_wrap": self.action_toggle_wrap,
            "editor.toggle_project": self.action_toggle_project,
            "preview.hide": self.action_hide_preview,
            "preview.refresh": self._refresh_preview,
            "preview.reset_camera": self.action_reset_camera,
            "preview.page_prev": self.action_preview_page_prev,
            "preview.page_next": self.action_preview_page_next,
        }
        handler = handlers.get(action)
        if handler is None:
            self.notify(f"Menu action not implemented: {action}")
            return
        handler()

    def on_screen_resume(self) -> None:
        self.query_one("#editor-tree", EditorProjectTree).reload()
        self.refresh_chat_pane()
        self._refresh_terminal_cwd()

    def _chat_log(self) -> RichLog | None:
        try:
            return self.query_one("#editor-aux-chat-log", RichLog)
        except NoMatches:
            return None

    def _write_chat_line(self, markup: str) -> None:
        log = self._chat_log()
        if log is not None:
            log.write(markup)

    def _flush_pending_chat(self) -> None:
        app = self.app
        pending_sys = getattr(app, "_pending_system_lines", None)
        if pending_sys:
            for text in list(pending_sys):
                self.append_chat_system(text)
            pending_sys.clear()
        pending_asst = getattr(app, "_pending_assistant_lines", None)
        if pending_asst:
            for text in list(pending_asst):
                self.append_chat_assistant(text)
            pending_asst.clear()

    def _replay_chat_history(self, state) -> None:
        log = self._chat_log()
        if log is None:
            return
        for msg in state.messages:
            role = str(msg.get("role", ""))
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "user":
                self._chat_turn += 1
                line = format_user_turn_line(content, turn=self._chat_turn, for_markup=True)
                log.write(line.markup)
            elif role == "assistant":
                header = format_assistant_turn_header(turn=self._chat_turn, for_markup=True)
                log.write(header.markup)
                for chunk in content.splitlines() or [""]:
                    log.write(format_assistant_body_line(chunk, for_markup=True).markup)
            elif role == "system":
                log.write(format_system_line(content, for_markup=True).markup)

    def refresh_chat_pane(self) -> None:
        try:
            status = self.query_one("#editor-aux-chat-status", Static)
            prompt = self.query_one("#editor-aux-chat-prompt", Input)
        except NoMatches:
            return
        app = self.app
        error = getattr(app, "session_error", "") or ""
        state = getattr(app, "session_state", None)
        if error:
            status.update(f"Chat · {error}")
            prompt.disabled = True
            return
        if state is None:
            load = getattr(app, "model_status_text", "") or getattr(app, "load_status", "loading...")
            status.update(f"Chat · {load}")
            prompt.disabled = True
            return
        if not self._chat_history_synced:
            self._replay_chat_history(state)
            self._flush_pending_chat()
            self._chat_history_synced = True
        label = model_identity_line(state)
        if self._chat_busy:
            status.update(f"Chat · {label} · generating")
            prompt.disabled = True
            return
        if not chat_session_ready(state):
            if is_server_backend(state.backend_id):
                status.update(f"Chat · {label} · pick a model with /models")
            else:
                status.update(f"Chat · {label} · no weights loaded")
            prompt.disabled = False
            return
        status.update(f"Chat · {label}")
        prompt.disabled = False

    def append_chat_system(self, text: str) -> None:
        for chunk in text.splitlines() or [""]:
            self._write_chat_line(format_system_line(chunk, for_markup=True).markup)

    def append_chat_assistant(self, text: str) -> None:
        self._write_chat_line(format_assistant_turn_header(turn=self._chat_turn, for_markup=True).markup)
        for chunk in text.splitlines() or [""]:
            self._write_chat_line(format_assistant_body_line(chunk, for_markup=True).markup)

    def _set_chat_busy(self, active: bool) -> None:
        self._chat_busy = active
        self.refresh_chat_pane()
        if not active:
            try:
                self.query_one("#editor-aux-chat-prompt", Input).focus()
            except NoMatches:
                pass

    def _refresh_terminal_cwd(self) -> None:
        try:
            label = self.query_one("#editor-aux-terminal-cwd", Static)
        except NoMatches:
            return
        cwd = Path(getattr(self.app, "cwd", self._root)).resolve()
        label.update(f"cwd · {cwd}")

    def append_terminal_log(self, text: str) -> None:
        self._refresh_terminal_cwd()
        try:
            log = self.query_one("#editor-aux-terminal-log", RichLog)
        except NoMatches:
            return
        for line in text.splitlines() or [""]:
            log.write(line)

    def _set_terminal_busy(self, active: bool) -> None:
        self._terminal_busy = active
        try:
            prompt = self.query_one("#editor-aux-terminal-prompt", Input)
        except NoMatches:
            return
        prompt.disabled = active
        if not active:
            prompt.focus()

    def _submit_terminal_command(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        event.input.value = ""
        if not raw or self._terminal_busy:
            return
        if raw.lower() in ("cls", "clear"):
            try:
                self.query_one("#editor-aux-terminal-log", RichLog).clear()
            except NoMatches:
                pass
            self._refresh_terminal_cwd()
            return
        if raw.lower() in ("exit", "quit"):
            self.append_terminal_log("Use Ctrl+Q to leave the TUI. This pane stays open.")
            return
        self.append_terminal_log(f"$ {raw}")
        self._set_terminal_busy(True)
        self.run_worker(
            lambda: self._terminal_worker(raw),
            thread=True,
            exclusive=True,
            group="editor-shell",
        )

    def _terminal_worker(self, command: str) -> None:
        app = self.app
        try:
            app.run_shell_command(command)
        finally:
            app.call_from_thread(self._set_terminal_busy, False)

    # TODO: interactive PTY / ConPTY so the pane can host a persistent shell
    # TODO: Ctrl+C should interrupt the running subprocess

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "editor-aux-terminal-prompt":
            self._submit_terminal_command(event)
            return
        if event.input.id != "editor-aux-chat-prompt":
            return
        app = self.app
        state = getattr(app, "session_state", None)
        if state is None or self._chat_busy:
            return
        line = event.value
        event.input.value = ""
        if not line.strip():
            return
        self._chat_turn += 1
        self._write_chat_line(format_user_turn_line(line, turn=self._chat_turn, for_markup=True).markup)
        self.run_worker(lambda: self._chat_turn_worker(line), thread=True, exclusive=True)

    def _chat_turn_worker(self, line: str) -> None:
        app = self.app
        state = getattr(app, "session_state", None)
        if state is None:
            return
        app.call_from_thread(self._set_chat_busy, True)
        try:
            handled, should_generate = dispatch_chat_line(state, line)
            if state.exit_requested:
                app.call_from_thread(app.exit)
                return
            if handled and not should_generate:
                return
            if not handled:
                state.messages.append({"role": "user", "content": line.rstrip()})
            run_chat_generation(state)
        finally:
            app.call_from_thread(self._set_chat_busy, False)

    def action_absorb_ctrl_c(self) -> None:
        self.notify(
            "Use the File menu on the project pane to open a folder. Keybinds live in .orodruin/keybinds.yaml.",
            timeout=2,
        )

    def action_refresh_tree(self) -> None:
        tree = self.query_one("#editor-tree", EditorProjectTree)
        tree.reload()
        self._reindex_project()
        self.notify(f"Workspace: {self._root}")

    def _set_files_collapsed(self, collapsed: bool) -> None:
        files = self.query_one("#editor-files-col", Vertical)
        split = self.query_one("#editor-files-split", PaneSplitter)
        rail = self.query_one("#editor-files-show", Button)
        if collapsed:
            width = files.size.width
            if width > 0:
                self._files_width = width
            files.display = False
            split.display = False
            rail.display = True
            self._files_collapsed = True
            return
        files.styles.width = self._files_width
        files.display = True
        split.display = True
        rail.display = False
        self._files_collapsed = False

    def action_toggle_project(self) -> None:
        self._set_files_collapsed(not self._files_collapsed)
        if self._files_collapsed:
            self.query_one("#editor-source", EditorSourceTextArea).focus()
            return
        self.query_one("#editor-tree", EditorProjectTree).focus()

    def _set_aux_collapsed(self, collapsed: bool) -> None:
        aux = self.query_one("#editor-aux", Vertical)
        split = self.query_one("#editor-aux-split", PaneSplitter)
        switch = self.query_one("#editor-aux-switch", ContentSwitcher)
        toggle = self.query_one("#editor-aux-toggle", Button)
        if collapsed:
            height = aux.size.height
            if height > 3:
                self._aux_height = height
            aux.add_class("-collapsed")
            aux.styles.height = 3
            switch.display = False
            split.display = False
            toggle.label = "Show"
            self._aux_collapsed = True
            return
        aux.remove_class("-collapsed")
        aux.styles.height = self._aux_height
        switch.display = True
        split.display = True
        toggle.label = "Hide"
        self._aux_collapsed = False

    def action_toggle_aux(self) -> None:
        self._set_aux_collapsed(not self._aux_collapsed)

    def _set_aux_tab(self, pane_id: str) -> None:
        self._aux_tab = pane_id
        switch = self.query_one("#editor-aux-switch", ContentSwitcher)
        switch.current = pane_id
        for button_id, target in _AUX_TABS:
            button = self.query_one(f"#{button_id}", Button)
            button.variant = "primary" if target == pane_id else "default"
        if self._aux_collapsed:
            self._set_aux_collapsed(False)
        if pane_id == "editor-aux-chat":
            self.refresh_chat_pane()
            try:
                self.query_one("#editor-aux-chat-prompt", Input).focus()
            except NoMatches:
                pass
            return
        if pane_id == "editor-aux-terminal":
            self._refresh_terminal_cwd()
            try:
                self.query_one("#editor-aux-terminal-prompt", Input).focus()
            except NoMatches:
                pass

    def action_toggle_preview(self) -> None:
        if self._preview_mode is None:
            self.notify("No preview for this file.")
            return
        self._preview_user_collapsed = not self._preview_user_collapsed
        self._set_preview_visible(True)
        self._sync_preview_toggle()

    def _sync_preview_toggle(self) -> None:
        return

    def on_pane_splitter_collapse_requested(self, event: PaneSplitter.CollapseRequested) -> None:
        splitter_id = event.splitter.id
        if splitter_id == "editor-files-split":
            self._set_files_collapsed(True)
            return
        if splitter_id == "editor-preview-split":
            if self._preview_mode is None:
                return
            self._preview_user_collapsed = True
            self._set_preview_visible(True)
            self._sync_preview_toggle()
            return
        if splitter_id == "editor-aux-split":
            self._set_aux_collapsed(True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.handle_nav_button(event.button.id):
            return
        button_id = event.button.id or ""
        if button_id == "editor-files-show":
            self._set_files_collapsed(False)
            self.query_one("#editor-tree", EditorProjectTree).focus()
            return
        if button_id == "editor-aux-toggle":
            self.action_toggle_aux()
            return
        for tab_button_id, pane_id in _AUX_TABS:
            if button_id == tab_button_id:
                self._set_aux_tab(pane_id)
                return

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        app = self.app
        app.cwd = Path(event.path).resolve()
        self._refresh_terminal_cwd()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = Path(event.path).resolve()
        if not _is_editable(path):
            return
        if self._current_path is not None and path == self._current_path:
            return
        if self._dirty:
            self._pending_switch = path
            self.app.push_screen(UnsavedSwitchModal(), self._on_unsaved_choice)
            return
        self._open_path(path)

    def _on_unsaved_choice(self, choice: str | None) -> None:
        if self._pending_close:
            self._pending_close = False
            if choice == "cancel" or choice is None:
                return
            if choice == "save":
                if not self._save_current():
                    return
            self._clear_editor()
            return
        target = self._pending_switch
        self._pending_switch = None
        if choice == "cancel" or choice is None or target is None:
            return
        if choice == "save":
            if not self._save_current():
                return
        self._open_path(target)

    def _end_ignore_changes(self) -> None:
        self._ignore_changes = False

    def _set_preview_visible(self, visible: bool) -> None:
        show = visible and not self._preview_user_collapsed
        preview_col = self.query_one("#editor-preview-col", Vertical)
        preview_split = self.query_one("#editor-preview-split", PaneSplitter)
        preview_col.display = show
        preview_split.display = show

    def _set_preview_mode(self, mode: str | None) -> None:
        self._preview_mode = mode
        self._set_preview_visible(mode is not None)
        self._sync_preview_toggle()
        self.query_one("#editor-preview-scroll", VerticalScroll).display = mode == "markdown"
        raster = self.query_one("#editor-raster", RasterPreview)
        raster.display = mode == "raster"
        if mode != "raster":
            raster.clear_preview()
        self.query_one("#editor-table", DataTable).display = mode == "table"
        notebook = self.query_one("#editor-notebook", NotebookPreview)
        notebook.display = mode == "notebook"
        if mode != "notebook":
            self.run_worker(notebook.clear_notebook, exclusive=True)

    def _kind_label(self, path: Path) -> str:
        if _is_python(path):
            return "Python"
        if _is_markdown(path):
            return "Markdown"
        if _is_svg(path):
            return "SVG"
        if _is_mesh(path):
            return "STL"
        if _is_notebook(path):
            return "Notebook"
        if _is_pdf(path):
            return "PDF"
        if path.suffix.lower() in {".yaml", ".yml"}:
            return "YAML"
        if path.suffix.lower() == ".toml":
            return "TOML"
        if _is_table(path):
            return path.suffix.lower().lstrip(".").upper()
        return "File"

    def _apply_editor_mode(self, path: Path) -> None:
        source = self.query_one("#editor-source", EditorSourceTextArea)
        language = _language_for(path)
        source.soft_wrap = (
            _is_markdown(path)
            or _is_svg(path)
            or _is_table(path)
            or _is_notebook(path)
            or _is_pdf(path)
        )
        source.read_only = self._source_locked
        source.set_completion_context(path, language)
        try:
            source.language = language
        except Exception as exc:
            source.language = None
            if language:
                self.notify(f"Syntax highlighting unavailable for {language}: {exc}", timeout=3)
        if _is_svg(path) or _is_mesh(path) or _is_pdf(path):
            self._set_preview_mode("raster")
            if _is_mesh(path):
                title = "Preview · arrows / drag orbit · r reset"
            elif _is_pdf(path):
                title = "Preview · PgUp/PgDn"
            else:
                title = "Preview"
        elif _is_markdown(path):
            self._set_preview_mode("markdown")
            title = "Preview"
        elif _is_table(path):
            self._set_preview_mode("table")
            title = "Preview"
        elif _is_notebook(path):
            self._set_preview_mode("notebook")
            title = "Preview · stored outputs only"
        else:
            self._set_preview_mode(None)
            title = "Preview"
        self._set_pane_title("preview", title)

    def _open_path(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            self.notify(f"Missing file: {resolved}")
            return
        if not _is_editable(resolved):
            self.notify(f"Unsupported file type: {resolved.suffix or resolved.name}")
            return
        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            self.notify(f"Could not read {resolved.name}: {exc}")
            return
        mesh: Mesh | None = None
        table: TableData | None = None
        pdf_bytes: bytes | None = None
        source_locked = False
        if _is_mesh(resolved):
            try:
                mesh = parse_stl_bytes(raw)
            except (OSError, ValueError) as exc:
                self.notify(f"Could not parse {resolved.name}: {exc}")
                return
            try:
                text = raw.decode("ascii")
                parse_stl_text(text)
            except (UnicodeDecodeError, ValueError):
                text = f"binary STL ({len(mesh.vertices)} facets)\nPreview is orbit-only. Saving is disabled."
                source_locked = True
        elif _is_binary_table(resolved):
            try:
                table = load_table(resolved, raw=raw)
            except (OSError, ValueError, KeyError) as exc:
                self.notify(f"Could not parse {resolved.name}: {exc}")
                return
            extra = f" · +{len(table.extra_sheets)} sheets" if table.extra_sheets else ""
            text = (
                f"xlsx · {table.sheet or 'Sheet1'} · {len(table.rows)} data rows{extra}\n"
                "Saving from the text editor is disabled."
            )
            source_locked = True
        elif _is_pdf(resolved):
            from cli.tui.pdf_view import open_pdf, pdf_source_note

            try:
                doc = open_pdf(raw)
                try:
                    text = pdf_source_note(doc)
                finally:
                    doc.close()
            except ValueError as exc:
                self.notify(f"Could not parse {resolved.name}: {exc}")
                return
            pdf_bytes = raw
            source_locked = True
        else:
            text = raw.decode("utf-8", errors="replace")
            if _is_table(resolved):
                try:
                    table = load_table(resolved, text=text, raw=raw)
                except (OSError, ValueError) as exc:
                    self.notify(f"Could not parse {resolved.name}: {exc}")
                    return
        self._source_locked = source_locked
        self._mesh = mesh
        self._table = table
        self._pdf_bytes = pdf_bytes
        self._ignore_changes = True
        source = self.query_one("#editor-source", EditorSourceTextArea)
        self._apply_editor_mode(resolved)
        source.load_text(text)
        self._current_path = resolved
        self._saved_text = text
        self._completer.set_buffer_text(text)
        try:
            rel = resolved.relative_to(self._root)
            label = str(rel).replace("\\", "/")
        except ValueError:
            label = str(resolved)
        kind = self._kind_label(resolved)
        self._set_pane_title("editor", f"{kind} · {label}")
        self._remember_recent(resolved)
        self._render_side_preview(text, reset_orbit=True)
        self.call_after_refresh(self._after_open)

    def _after_open(self) -> None:
        self._end_ignore_changes()
        path = self._current_path
        if path is None:
            return
        if _is_mesh(path) or _is_svg(path) or _is_pdf(path):
            raster = self.query_one("#editor-raster", RasterPreview)
            raster.repaint()
            raster.focus()
            if _is_mesh(path):
                self.notify("STL: arrows or drag to orbit. r resets.", timeout=4)
            elif _is_pdf(path):
                self.notify("PDF: PgUp/PgDn or left/right changes page. Preview-only.", timeout=4)
            return
        if _is_table(path):
            self.query_one("#editor-table", DataTable).focus()
            return
        if _is_notebook(path):
            self.query_one("#editor-notebook", NotebookPreview).focus()
            return
        self.query_one("#editor-source", EditorSourceTextArea).focus()

    def on_raster_preview_pdf_page_changed(self, event: RasterPreview.PdfPageChanged) -> None:
        path = self._current_path
        if path is None or not _is_pdf(path):
            return
        self._set_pane_title("preview", f"Preview · page {event.index + 1}/{event.count} · PgUp/PgDn")

    def _render_side_preview(self, text: str, *, reset_orbit: bool = False) -> None:
        path = self._current_path
        if path is None:
            return
        if _is_markdown(path):
            preview = self.query_one("#editor-preview", Markdown)
            preview.update(text or "_Empty file._")
            scroll = self.query_one("#editor-preview-scroll", VerticalScroll)
            scroll.scroll_home(animate=False)
            return
        raster = self.query_one("#editor-raster", RasterPreview)
        if _is_svg(path):
            try:
                raster.show_svg(text)
            except Exception as exc:
                raster.show_error(f"SVG preview failed: {exc}")
            return
        if _is_mesh(path):
            mesh = self._mesh
            if not self._source_locked:
                try:
                    mesh = parse_stl_text(text)
                    self._mesh = mesh
                except ValueError as exc:
                    raster.show_error(f"STL preview failed: {exc}")
                    return
            if mesh is None:
                raster.show_error("STL preview failed: no mesh.")
                return
            raster.show_mesh(mesh, reset_orbit=reset_orbit)
            return
        if _is_pdf(path):
            data = self._pdf_bytes
            if not data:
                raster.show_error("PDF preview failed: no file bytes.")
                return
            raster.show_pdf(data)
            return
        if _is_table(path):
            try:
                if self._source_locked and self._table is not None:
                    table = self._table
                else:
                    table = load_table(path, text=text)
                    self._table = table
            except (OSError, ValueError, KeyError) as exc:
                self._show_table_error(str(exc))
                return
            self._show_table(table)
            return
        if _is_notebook(path):
            notebook = self.query_one("#editor-notebook", NotebookPreview)
            self.run_worker(notebook.show_notebook(text), exclusive=True)

    def _show_table(self, table: TableData) -> None:
        widget = self.query_one("#editor-table", DataTable)
        widget.clear(columns=True)
        headers = table.headers or ["(empty)"]
        widget.add_columns(*headers)
        if table.rows:
            widget.add_rows(table.rows)
        bits = [f"{len(table.rows)} rows", f"{len(headers)} cols"]
        if table.sheet:
            bits.insert(0, table.sheet)
        if table.truncated:
            bits.append("truncated")
        if table.extra_sheets:
            bits.append(f"+{len(table.extra_sheets)} sheets")
        self._set_pane_title("preview", "Preview · " + " · ".join(bits))

    def _show_table_error(self, message: str) -> None:
        widget = self.query_one("#editor-table", DataTable)
        widget.clear(columns=True)
        widget.add_columns("error")
        widget.add_rows([[message]])
        self._set_pane_title("preview", "Preview · parse failed")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "editor-source":
            return
        if self._ignore_changes:
            return
        text = event.text_area.text
        self._completer.set_buffer_text(text)
        if self._source_locked or self._current_path is None:
            return
        if not (
            _is_markdown(self._current_path)
            or _is_svg(self._current_path)
            or _is_mesh(self._current_path)
            or (_is_table(self._current_path) and not self._source_locked)
            or _is_notebook(self._current_path)
        ):
            return
        if self._preview_timer is not None:
            self._preview_timer.stop()
        self._preview_timer = self.set_timer(_PREVIEW_DEBOUNCE_S, self._refresh_preview)

    def _refresh_preview(self) -> None:
        self._preview_timer = None
        if self._current_path is None:
            return
        text = self.query_one("#editor-source", EditorSourceTextArea).text
        self._render_side_preview(text)

    def _save_current(self) -> bool:
        path = self._current_path
        if path is None:
            self.notify("No file selected.")
            return False
        if self._source_locked:
            self.notify("This file cannot be saved from the text editor.")
            return False
        text = self.query_one("#editor-source", EditorSourceTextArea).text
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            self.notify(f"Save failed: {exc}")
            return False
        self._saved_text = text
        self._reindex_project()
        if path.name == "keybinds.yaml" and path.parent.name == ".orodruin":
            self._apply_keybinds()
        if _is_markdown(path) or _is_svg(path) or _is_mesh(path) or _is_table(path) or _is_notebook(path):
            self._refresh_preview()
        self.notify(f"Saved {path.name}")
        return True

    def action_save_file(self) -> None:
        if self._current_path is None:
            self.action_save_as()
            return
        self._save_current()

    def action_save_as(self) -> None:
        current = self._current_path
        start = current.parent if current is not None else self._target_directory()
        initial = str(current) if current is not None else str(start / "untitled.md")
        self.app.push_screen(
            PathPickerModal(
                mode="save",
                start=start,
                title="Save As",
                confirm_label="Save",
                initial=initial,
            ),
            self._on_save_as_path,
        )

    def action_new_file(self) -> None:
        target = self._target_directory()
        self.app.push_screen(
            NamePromptModal(
                f"New file in {target.name}",
                placeholder="untitled.md",
                confirm_label="Create",
            ),
            lambda name: self._create_file(target, name),
        )

    def action_new_folder(self) -> None:
        target = self._target_directory()
        self.app.push_screen(
            NamePromptModal(
                f"New folder in {target.name}",
                placeholder="folder name",
                confirm_label="Create",
            ),
            lambda name: self._create_folder(target, name),
        )

    def action_open_file(self) -> None:
        start = self._current_path.parent if self._current_path is not None else self._root
        self.app.push_screen(
            PathPickerModal(
                mode="file",
                start=start,
                title="Open File",
                confirm_label="Open",
                initial=str(start),
            ),
            self._request_open_path,
        )

    def action_open_folder(self) -> None:
        self.app.push_screen(
            PathPickerModal(
                mode="folder",
                start=self._root,
                title="Open Folder",
                confirm_label="Open",
                initial=str(self._root),
            ),
            self._on_open_folder,
        )

    def action_edit_keybinds(self) -> None:
        workspace = self._root
        app_root = self._profile_root()
        if (workspace / ".orodruin" / "keybinds.yaml").is_file():
            path = keybinds_path(workspace)
        else:
            path = ensure_keybinds_file(app_root)
        self._request_open_path(path)

    def action_reload_keybinds(self) -> None:
        self._apply_keybinds()
        self.notify("Reloaded keybinds from .orodruin/keybinds.yaml")

    def action_close_file(self) -> None:
        if self._current_path is None:
            self.notify("No file selected.")
            return
        if self._dirty:
            self._pending_close = True
            self.app.push_screen(UnsavedSwitchModal(), self._on_unsaved_choice)
            return
        self._clear_editor()

    def action_revert_file(self) -> None:
        path = self._current_path
        if path is None:
            self.notify("No file selected.")
            return
        self._open_path(path)

    def action_find_in_file(self) -> None:
        self.app.push_screen(
            NamePromptModal(
                "Find",
                placeholder="Search text",
                confirm_label="Find",
            ),
            self._find_in_source,
        )

    def action_collapse_all(self) -> None:
        tree = self.query_one("#editor-tree", EditorProjectTree)
        for node in tree.root.children:
            if getattr(node, "is_expanded", False):
                node.collapse()

    def action_toggle_line_numbers(self) -> None:
        source = self.query_one("#editor-source", EditorSourceTextArea)
        source.show_line_numbers = not source.show_line_numbers

    def action_toggle_wrap(self) -> None:
        source = self.query_one("#editor-source", EditorSourceTextArea)
        source.soft_wrap = not source.soft_wrap

    def action_hide_preview(self) -> None:
        if self._preview_mode is None:
            return
        self._preview_user_collapsed = True
        self._set_preview_visible(True)
        self._sync_preview_toggle()

    def action_reset_camera(self) -> None:
        raster = self.query_one("#editor-raster", RasterPreview)
        raster.action_reset_orbit()

    def action_preview_page_prev(self) -> None:
        raster = self.query_one("#editor-raster", RasterPreview)
        raster.action_page_prev()

    def action_preview_page_next(self) -> None:
        raster = self.query_one("#editor-raster", RasterPreview)
        raster.action_page_next()

    def _run_source_action(self, name: str) -> None:
        source = self.query_one("#editor-source", EditorSourceTextArea)
        source.focus()
        method = getattr(source, name, None)
        if method is None:
            self.notify(f"{name} is not available.")
            return
        method()

    def _remember_recent(self, path: Path) -> None:
        resolved = path.resolve()
        self._recent_paths = [item for item in self._recent_paths if item != resolved]
        self._recent_paths.insert(0, resolved)
        del self._recent_paths[8:]

    def _target_directory(self) -> Path:
        tree = self.query_one("#editor-tree", EditorProjectTree)
        node = getattr(tree, "cursor_node", None)
        data = getattr(node, "data", None) if node is not None else None
        if data is None:
            return self._root
        path = Path(data.path)
        if path.is_dir():
            return path
        parent = path.parent
        return parent if parent.exists() else self._root

    def _safe_join(self, base: Path, name: str) -> Path | None:
        cleaned = name.strip().replace("\\", "/")
        if not cleaned or cleaned.startswith("/"):
            self.notify("Enter a relative name.")
            return None
        if ".." in Path(cleaned).parts:
            self.notify("Path must stay inside the target folder.")
            return None
        return (base / cleaned).resolve()

    def _set_workspace(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_dir():
            self.notify(f"Not a folder: {resolved}")
            return
        self._root = resolved
        self.app.cwd = resolved
        tree = self.query_one("#editor-tree", EditorProjectTree)
        try:
            tree.path = resolved
        except Exception:
            pass
        tree.reload()
        self._remember_recent(resolved)
        self._reindex_project()
        self._set_pane_title("project", f"Project · {resolved.name}")
        self.notify(f"Opened folder {resolved}")

    def _request_open_path(self, path: Path | None) -> None:
        if path is None:
            return
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            self._set_workspace(resolved)
            return
        if self._current_path is not None and resolved == self._current_path:
            return
        if self._dirty:
            self._pending_switch = resolved
            self.app.push_screen(UnsavedSwitchModal(), self._on_unsaved_choice)
            return
        self._open_path(resolved)

    def _on_open_folder(self, path: Path | None) -> None:
        if path is None:
            return
        self._set_workspace(path)

    def _create_file(self, target: Path, name: str | None) -> None:
        if not name:
            return
        path = self._safe_join(target, name)
        if path is None:
            return
        if path.exists():
            self.notify(f"Already exists: {path.name}")
            if path.is_file():
                self._request_open_path(path)
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        except OSError as exc:
            self.notify(f"Could not create {path.name}: {exc}")
            return
        self.action_refresh_tree()
        self._request_open_path(path)

    def _create_folder(self, target: Path, name: str | None) -> None:
        if not name:
            return
        path = self._safe_join(target, name)
        if path is None:
            return
        if path.exists():
            self.notify(f"Already exists: {path.name}")
            return
        try:
            path.mkdir(parents=True)
        except OSError as exc:
            self.notify(f"Could not create folder: {exc}")
            return
        self.action_refresh_tree()
        self.notify(f"Created folder {path.name}")

    def _on_save_as_path(self, path: Path | None) -> None:
        if path is None:
            return
        resolved = path.expanduser()
        try:
            resolved = resolved.resolve()
        except OSError as exc:
            self.notify(f"Save failed: {exc}")
            return
        if resolved.is_dir():
            self.notify("Choose a file path, not a folder.")
            return
        if self._source_locked and self._current_path is not None and resolved == self._current_path:
            self.notify("This file cannot be saved from the text editor.")
            return
        text = self.query_one("#editor-source", EditorSourceTextArea).text
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(text, encoding="utf-8")
        except OSError as exc:
            self.notify(f"Save failed: {exc}")
            return
        self._source_locked = False
        self._current_path = resolved
        self._saved_text = text
        self._apply_editor_mode(resolved)
        try:
            rel = resolved.relative_to(self._root)
            label = str(rel).replace("\\", "/")
        except ValueError:
            label = str(resolved)
        kind = self._kind_label(resolved)
        self._set_pane_title("editor", f"{kind} · {label}")
        self._remember_recent(resolved)
        self._reindex_project()
        self.action_refresh_tree()
        self.notify(f"Saved {resolved.name}")

    def _clear_editor(self) -> None:
        self._ignore_changes = True
        self._current_path = None
        self._saved_text = ""
        self._source_locked = False
        self._mesh = None
        self._table = None
        self._pdf_bytes = None
        source = self.query_one("#editor-source", EditorSourceTextArea)
        source.load_text("")
        source.language = None
        source.read_only = False
        source.set_completion_context(None, None)
        self._set_pane_title("editor", "Editor")
        self._set_preview_mode(None)
        self.call_after_refresh(self._end_ignore_changes)

    def _find_in_source(self, query: str | None) -> None:
        if not query:
            return
        source = self.query_one("#editor-source", EditorSourceTextArea)
        text = source.text
        row, col = source.cursor_location
        lines = text.split("\n")
        start = sum(len(line) + 1 for line in lines[:row]) + col
        pos = text.find(query, start + 1)
        wrapped = False
        if pos < 0:
            pos = text.find(query)
            wrapped = True
        if pos < 0:
            self.notify(f"Not found: {query}")
            return
        before = text[:pos]
        new_row = before.count("\n")
        last_nl = before.rfind("\n")
        new_col = pos if last_nl < 0 else pos - (last_nl + 1)
        source.cursor_location = (new_row, new_col)
        source.focus()
        scroller = getattr(source, "scroll_cursor_visible", None)
        if scroller is not None:
            scroller()
        if wrapped:
            self.notify("Reached the start of the file.")

    def action_accept_completion(self) -> None:
        try:
            source = self.query_one("#editor-source", EditorSourceTextArea)
        except NoMatches:
            return
        if not source.has_focus or not source.suggestion:
            return
        source.insert(source.suggestion)
