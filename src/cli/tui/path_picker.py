from __future__ import annotations

import os
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Static

_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "$recycle.bin",
    "system volume information",
}
_SKIP_WSL_DISTROS = {
    "docker-desktop",
    "docker-desktop-data",
}


def disk_roots() -> list[Path]:
    roots: list[Path] = []
    if os.name == "nt" or sys.platform == "win32":
        for letter in string.ascii_uppercase:
            candidate = Path(f"{letter}:/")
            try:
                if candidate.exists():
                    roots.append(candidate)
            except OSError:
                continue
        return roots
    roots.append(Path("/"))
    mnt = Path("/mnt")
    try:
        if mnt.is_dir():
            for child in sorted(mnt.iterdir()):
                if child.is_dir() and len(child.name) == 1 and child.name.isalpha():
                    roots.append(child)
    except OSError:
        pass
    return roots


def _drive_id(root: Path) -> str:
    if root.drive:
        return root.drive[0].upper()
    stripped = str(root).replace("\\", "/").strip("/")
    return stripped.replace("/", "-") or "root"


def _drive_label(root: Path) -> str:
    if root.drive:
        return f"{root.drive[0].upper()}:"
    return str(root)


def _is_fs_root(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return resolved.parent == resolved


def _decode_wsl_output(raw: bytes) -> str:
    if not raw:
        return ""
    sample = raw[: min(32, len(raw))]
    if raw.startswith(b"\xff\xfe") or b"\x00" in sample:
        return raw.decode("utf-16-le", errors="replace").replace("\x00", "")
    return raw.decode("utf-8", errors="replace")


def _wsl_exe() -> str | None:
    if sys.platform == "win32" or os.name == "nt":
        return shutil.which("wsl") or shutil.which("wsl.exe")
    for candidate in ("wsl.exe", "/mnt/c/Windows/System32/wsl.exe"):
        located = shutil.which(candidate)
        if located:
            return located
        if Path(candidate).is_file():
            return candidate
    return None


def _wsl_key(name: str) -> str:
    chars = [ch if ch.isalnum() else "-" for ch in name.strip()]
    key = "".join(chars).strip("-").lower()
    return key or "distro"


def _wsl_distro_home(exe: str, distro: str) -> Path | None:
    current = os.environ.get("WSL_DISTRO_NAME", "").strip()
    try:
        proc = subprocess.run(
            [exe, "-d", distro, "-e", "printenv", "HOME"],
            capture_output=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _wsl_distro_root(distro)
    lines = _decode_wsl_output(proc.stdout).strip().splitlines()
    home = lines[0].strip() if lines else ""
    if not home.startswith("/"):
        return _wsl_distro_root(distro)
    if current and current.lower() == distro.lower():
        local = Path(home)
        try:
            if local.is_dir():
                return local
        except OSError:
            pass
    if sys.platform == "win32" or os.name == "nt":
        rel = home.lstrip("/").replace("/", "\\")
        for prefix in (rf"\\wsl.localhost\{distro}", rf"\\wsl$\{distro}"):
            candidate = Path(prefix + "\\" + rel)
            try:
                if candidate.is_dir():
                    return candidate
            except OSError:
                continue
        return Path(rf"\\wsl.localhost\{distro}\{rel}")
    mounted = Path("/mnt/wsl") / distro
    nested = Path(str(mounted) + home)
    try:
        if nested.is_dir():
            return nested
    except OSError:
        pass
    return _wsl_distro_root(distro)


def _wsl_distro_root(distro: str) -> Path | None:
    if sys.platform == "win32" or os.name == "nt":
        return Path(rf"\\wsl.localhost\{distro}")
    current = os.environ.get("WSL_DISTRO_NAME", "").strip()
    if current and current.lower() == distro.lower():
        return Path("/")
    mounted = Path("/mnt/wsl") / distro
    try:
        if mounted.is_dir():
            return mounted
    except OSError:
        pass
    return None


def wsl_home_targets() -> list[tuple[str, str, Path]]:
    exe = _wsl_exe()
    if exe is None:
        return []
    try:
        proc = subprocess.run([exe, "-l", "-q"], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    targets: list[tuple[str, str, Path]] = []
    used_keys: set[str] = set()
    for line in _decode_wsl_output(proc.stdout).splitlines():
        name = line.strip().strip("\ufeff")
        if not name or name.lower() in _SKIP_WSL_DISTROS:
            continue
        home = _wsl_distro_home(exe, name)
        if home is None:
            continue
        key = _wsl_key(name)
        suffix = 2
        unique = key
        while unique in used_keys:
            unique = f"{key}-{suffix}"
            suffix += 1
        used_keys.add(unique)
        targets.append((unique, name, home))
    return targets


def _as_existing_dir(path: Path) -> Path | None:
    try:
        expanded = path.expanduser()
    except OSError:
        return None
    try:
        resolved = expanded.resolve()
        if resolved.is_dir():
            return resolved
        if resolved.is_file():
            return resolved.parent
    except OSError:
        pass
    try:
        if expanded.is_dir():
            return expanded
        if expanded.is_file():
            return expanded.parent
    except OSError:
        return None
    return None


class PickerEnterDirectory(Message):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__()


class PickerTree(DirectoryTree):
    auto_expand = False

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        # TODO: optional native OS file dialog instead of this tree picker
        kept: list[Path] = []
        for path in paths:
            if path.name.lower() in _SKIP_DIR_NAMES:
                continue
            kept.append(path)
        return kept

    async def _on_click(self, event: events.Click) -> None:
        if event.chain != 2:
            return
        meta = event.style.meta
        line_no = meta.get("line") if meta else None
        node = self.get_node_at_line(line_no) if isinstance(line_no, int) else None
        if node is None or node.data is None:
            return
        path = Path(node.data.path)
        try:
            is_directory = path.is_dir()
        except OSError:
            return
        if not is_directory:
            return
        entered = _as_existing_dir(path) or path
        self.post_message(PickerEnterDirectory(entered))
        event.stop()


class PathPickerModal(ModalScreen[Path | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("backspace", "go_up", "Up", show=False),
    ]

    DEFAULT_CSS = """
    PathPickerModal {
        align: center middle;
        background: $background 60%;
    }
    """

    def __init__(
        self,
        *,
        mode: str,
        start: Path,
        title: str,
        confirm_label: str = "Open",
        initial: str = "",
    ) -> None:
        super().__init__()
        self._mode = mode
        try:
            self._start = start if start.is_dir() else start.parent
        except OSError:
            self._start = start
        self._title = title
        self._confirm_label = confirm_label
        self._initial = initial
        self._browse_root = self._start
        self._drives = {_drive_id(root): root for root in disk_roots()}
        wsl_targets = wsl_home_targets()
        self._wsl_homes = {key: home for key, _name, home in wsl_targets}
        self._wsl_labels = {key: name for key, name, _home in wsl_targets}

    def compose(self) -> ComposeResult:
        with Vertical(id="editor-path-dialog"):
            yield Static(self._title, id="editor-path-title")
            with Horizontal(id="editor-path-nav"):
                yield Button("Up", id="editor-path-up", variant="default")
                yield Button("Home", id="editor-path-home", variant="default")
                yield Static("", id="editor-path-nav-spacer")
                for drive_id, root in self._drives.items():
                    yield Button(
                        _drive_label(root),
                        id=f"editor-path-drive-{drive_id}",
                        variant="default",
                    )
            if self._wsl_homes:
                with Horizontal(id="editor-path-wsl"):
                    yield Static("WSL", classes="editor-path-nav-label")
                    for key, name in self._wsl_labels.items():
                        yield Button(
                            name,
                            id=f"editor-path-wsl-{key}",
                            variant="default",
                        )
            yield PickerTree(str(self._start), id="editor-path-tree")
            with Horizontal(id="editor-path-entry"):
                yield Input(
                    value=self._initial or str(self._start),
                    placeholder="Path",
                    id="editor-path-input",
                )
                yield Button("Go", id="editor-path-go", variant="default")
            with Horizontal(id="editor-path-actions"):
                yield Button(self._confirm_label, id="editor-path-open", variant="primary")
                yield Button("Cancel", id="editor-path-cancel", variant="default")

    def on_mount(self) -> None:
        self._browse(self._start)
        self.query_one("#editor-path-tree", PickerTree).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_go_up(self) -> None:
        self._go_up()

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        path = Path(event.path)
        directory = _as_existing_dir(path) or path
        self.query_one("#editor-path-input", Input).value = str(directory)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = Path(event.path)
        try:
            path = path.resolve()
        except OSError:
            pass
        self.query_one("#editor-path-input", Input).value = str(path)

    def on_picker_enter_directory(self, event: PickerEnterDirectory) -> None:
        self._browse(event.path)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "editor-path-input":
            self._go_input_path()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "editor-path-cancel":
            self.dismiss(None)
            return
        if button_id == "editor-path-open":
            self._confirm()
            return
        if button_id == "editor-path-up":
            self._go_up()
            return
        if button_id == "editor-path-home":
            self._browse(Path.home())
            return
        if button_id == "editor-path-go":
            self._go_input_path()
            return
        prefix = "editor-path-drive-"
        if button_id.startswith(prefix):
            drive_id = button_id[len(prefix) :]
            root = self._drives.get(drive_id)
            if root is not None:
                self._browse(root)
            return
        wsl_prefix = "editor-path-wsl-"
        if button_id.startswith(wsl_prefix):
            key = button_id[len(wsl_prefix) :]
            home = self._wsl_homes.get(key)
            if home is not None:
                self._browse(home)

    def _go_up(self) -> None:
        if _is_fs_root(self._browse_root):
            return
        parent = self._browse_root.parent
        if parent == self._browse_root:
            return
        self._browse(parent)

    def _go_input_path(self) -> None:
        raw = self.query_one("#editor-path-input", Input).value.strip()
        if not raw:
            return
        location = _as_existing_dir(Path(raw))
        if location is None:
            return
        self._browse(location)

    def _browse(self, location: Path) -> None:
        directory = _as_existing_dir(location)
        if directory is None:
            return
        self._browse_root = directory
        tree = self.query_one("#editor-path-tree", PickerTree)
        try:
            tree.path = directory
        except Exception:
            pass
        tree.reload()
        self.query_one("#editor-path-input", Input).value = str(directory)
        title = f"{self._title} · {directory}"
        self.query_one("#editor-path-title", Static).update(title)
        up = self.query_one("#editor-path-up", Button)
        up.disabled = _is_fs_root(directory)

    def _confirm(self) -> None:
        raw = self.query_one("#editor-path-input", Input).value.strip()
        if not raw:
            return
        path = Path(raw).expanduser()
        try:
            path = path.resolve()
        except OSError:
            pass
        if self._mode == "folder":
            directory = _as_existing_dir(path)
            if directory is None:
                return
            self.dismiss(directory)
            return
        try:
            is_directory = path.is_dir()
            is_file = path.is_file()
            parent_ok = path.parent.is_dir()
        except OSError:
            return
        if is_directory:
            return
        if self._mode == "save":
            if not parent_ok:
                return
            self.dismiss(path)
            return
        if not is_file:
            return
        self.dismiss(path)


class NamePromptModal(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    DEFAULT_CSS = """
    NamePromptModal {
        align: center middle;
        background: $background 60%;
    }
    """

    def __init__(
        self,
        title: str,
        placeholder: str = "",
        initial: str = "",
        confirm_label: str = "Create",
    ) -> None:
        super().__init__()
        self._title = title
        self._placeholder = placeholder
        self._initial = initial
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="editor-name-dialog"):
            yield Static(self._title, id="editor-name-title")
            yield Input(
                value=self._initial,
                placeholder=self._placeholder,
                id="editor-name-input",
            )
            with Horizontal(id="editor-name-actions"):
                yield Button(self._confirm_label, id="editor-name-ok", variant="primary")
                yield Button("Cancel", id="editor-name-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#editor-name-input", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "editor-name-input":
            self._confirm()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "editor-name-cancel":
            self.dismiss(None)
            return
        if event.button.id == "editor-name-ok":
            self._confirm()

    def _confirm(self) -> None:
        value = self.query_one("#editor-name-input", Input).value.strip()
        if not value:
            return
        self.dismiss(value)
