from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from utils.device.env_bootstrap import sophon_project_root

PROFILE_DIRNAME = ".sophon"
KEYBINDS_FILENAME = "keybinds.yaml"

DEFAULT_KEYBINDS: dict[str, str] = {
    "save_file": "ctrl+s",
    "save_as": "ctrl+shift+s",
    "new_file": "ctrl+n",
    "new_folder": "",
    "open_file": "ctrl+o",
    "open_folder": "ctrl+shift+o",
    "find_in_file": "ctrl+f",
    "refresh_tree": "ctrl+r",
    "cycle_tree": "ctrl+shift+t",
    "toggle_project": "ctrl+b",
    "toggle_aux": "ctrl+j",
    "accept_completion": "right",
    "copy_chat": "ctrl+y",
    "listen": "ctrl+l",
    "open_models": "ctrl+m",
    "open_dashboard": "ctrl+d",
    "open_editor": "ctrl+e",
    "open_chat": "ctrl+g",
    "cycle_mode": "ctrl+h",
    "permission_once": "1",
    "permission_persist": "2",
    "permission_deny": "3",
    "quit": "",
    "help": "",
    "focus_prompt": "",
    "focus_transcript": "",
    "accept_edits": "",
    "decline_edits": "",
    "energy_local": "",
    "energy_api": "",
    "compact_now": "",
    "tools": "",
    "search": "",
}

NAV_ACTIONS = ("open_dashboard", "open_editor", "open_chat", "cycle_mode")
PANE_ACTIONS = ("open_dashboard", "open_editor", "open_chat")
PERMISSION_ACTIONS = ("permission_once", "permission_persist", "permission_deny")

DEFAULT_SLASH_BINDS: dict[str, str] = {
    "models": "ctrl+shift+m",
    "permissions": "f2",
}

CHAT_KEYBIND_ACTIONS = frozenset({"copy_chat", "listen", "open_models"})

# Repo-local (workspace) keybinds files may only rebind these editor-local actions.
WORKSPACE_ACTIONS = frozenset(
    {
        "save_file",
        "save_as",
        "new_file",
        "new_folder",
        "open_file",
        "open_folder",
        "find_in_file",
        "toggle_project",
        "toggle_aux",
        "accept_completion",
    }
)
MAX_KEYBINDS_BYTES = 65_536
MAX_REPORTED = 50
# One Textual key: up to three modifiers and a key token. Always used with fullmatch.
KEY_PATTERN = re.compile(r"(?:(?:ctrl|shift|alt|meta|super)\+){0,3}[A-Za-z0-9_]{1,20}")
WORKSPACE_KEY_OK = re.compile(
    r"(?:(?:ctrl|shift|alt|meta|super)\+){1,3}[A-Za-z0-9_]{1,20}|f(?:[1-9]|1[0-9]|2[0-4])"
)
WORKSPACE_RESERVED = frozenset({"ctrl+q", "ctrl+c", "ctrl+p", "tab", "shift+tab", "f1", "super+c"})
# Characters that would break the `run_slash('...')` action string.
PROFILE_SLASH_BAD = re.compile(r"['\"()\\\x00-\x1f\x7f]")
MAX_SLASH_NAME = 128
_NULL_VALUES = frozenset({"", "~", "null", "Null", "NULL"})
_SAFE_LABEL = re.compile(r"[A-Za-z0-9_\-]{1,40}")

_KEYBINDS_TEMPLATE = """# sophon keybinds (project profile)
# Restart is not required if you save this file from the editor.
# Textual key names: ctrl+s, ctrl+shift+o, right, f2
# Empty quotes unbind an action.
# slash: maps a key to /command (name without the slash).

save_file: ctrl+s
save_as: ctrl+shift+s
new_file: ctrl+n
new_folder: ""
open_file: ctrl+o
open_folder: ctrl+shift+o
find_in_file: ctrl+f
refresh_tree: ctrl+r
cycle_tree: ctrl+shift+t
toggle_project: ctrl+b
toggle_aux: ctrl+j
accept_completion: right
copy_chat: ctrl+y
listen: ctrl+l
open_models: ctrl+m
open_dashboard: ctrl+d
open_editor: ctrl+e
open_chat: ctrl+g
cycle_mode: ctrl+h
permission_once: "1"
permission_persist: "2"
permission_deny: "3"
quit: ""
help: ""
focus_prompt: ""
focus_transcript: ""
accept_edits: ""
decline_edits: ""
energy_local: ""
energy_api: ""
compact_now: ""
tools: ""
search: ""
slash:
  models: ctrl+shift+m
  permissions: f2
"""


@dataclass
class KeybindSet:
    actions: dict[str, str] = field(default_factory=dict)
    slash: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RawKeybinds:
    """Entries present in one keybinds file, validated but not merged with defaults."""

    actions: dict[str, str]
    slash: dict[str, str]
    problems: tuple[str, ...]


@dataclass(frozen=True)
class KeybindLoad:
    """The effective set for a workspace plus every workspace entry that was not applied."""

    binds: KeybindSet
    ignored: tuple[tuple[str, str], ...]
    source: Path | None


class _AliasRefused(yaml.YAMLError):
    pass


class _NoAliasLoader(yaml.BaseLoader):
    """Pure-Python loader: every scalar stays a string, and aliases are refused.

    BaseLoader avoids typed construction (timestamps, huge ints) that can raise; refusing
    aliases stops expansion bombs. Never swap in CBaseLoader: it bypasses compose_node.
    """

    def compose_node(self, parent: object, index: object) -> object:
        if self.check_event(yaml.events.AliasEvent):
            raise _AliasRefused("aliases are not allowed")
        return super().compose_node(parent, index)


class SlashDispatchMixin:
    def action_run_slash(self, name: str) -> None:
        from cli.chat import dispatch_chat_line

        cmd_name = str(name or "").strip().lstrip("/")
        if not cmd_name:
            return
        if cmd_name == "models":
            open_models = getattr(self, "action_open_models", None)
            if callable(open_models):
                open_models()
                return
        state = getattr(self.app, "session_state", None)
        if state is None:
            return
        dispatch_chat_line(state, "/" + cmd_name)
        refresh = getattr(self, "refresh_from_state", None)
        if not callable(refresh):
            refresh = getattr(self, "refresh_chat_pane", None)
        if callable(refresh):
            refresh()


def profile_dir(root: Path | None = None) -> Path:
    return (root or sophon_project_root()) / PROFILE_DIRNAME


def keybinds_path(root: Path | None = None) -> Path:
    return profile_dir(root) / KEYBINDS_FILENAME


def format_key_label(key: str) -> str:
    raw = (key or "").strip()
    if not raw:
        return ""
    parts = [part.strip() for part in raw.split("+") if part.strip()]
    pretty: list[str] = []
    for part in parts:
        lower = part.lower()
        if lower == "ctrl":
            pretty.append("Ctrl")
        elif lower == "shift":
            pretty.append("Shift")
        elif lower == "alt":
            pretty.append("Alt")
        else:
            pretty.append(part.upper() if len(part) == 1 else part.capitalize())
    return "+".join(pretty)


def format_key_compact(key: str) -> str:
    raw = (key or "").strip()
    if not raw:
        return ""
    parts = [part.strip().lower() for part in raw.split("+") if part.strip()]
    mods = [part for part in parts if part in {"ctrl", "shift", "alt"}]
    rest = [part for part in parts if part not in {"ctrl", "shift", "alt"}]
    token = rest[-1] if rest else ""
    if mods == ["ctrl"] and len(token) == 1:
        return "^" + token
    return format_key_label(raw)


def panes_footer_label(binds: KeybindSet | None = None) -> str:
    current = binds or load_keybinds()
    bits = [
        format_key_compact(current.actions.get(action, "")) or "-"
        for action in PANE_ACTIONS
    ]
    return "/".join(bits) + " Panes"


def nav_chord_keys(binds: KeybindSet | None = None) -> frozenset[str]:
    current = binds or load_keybinds()
    keys = []
    for action in NAV_ACTIONS:
        key = str(current.actions.get(action, "") or "").strip().lower()
        if key:
            keys.append(key)
    return frozenset(keys)


def permission_keys(binds: KeybindSet | None = None) -> dict[str, str]:
    current = binds or load_keybinds()
    out: dict[str, str] = {}
    for action in PERMISSION_ACTIONS:
        out[action] = str(current.actions.get(action, "") or "").strip()
    return out


def _header() -> str:
    return (
        "# sophon keybinds (project profile)\n"
        "# Restart is not required if you save this file from the editor or /keybind set.\n"
        "# Textual key names: ctrl+s, ctrl+shift+o, right, f2\n"
        "# Empty quotes unbind an action.\n"
        "# slash: maps a key to /command (name without the slash).\n"
        "# TODO: quit, help, focus_prompt, focus_transcript, accept_edits, decline_edits,\n"
        "# energy_local, energy_api, compact_now, tools, and search have no handler yet.\n\n"
    )


def key_owner(binds: KeybindSet, key: str, *, ignore_action: str = "") -> str:
    token = key.strip().lower()
    if not token:
        return ""
    for action, existing in binds.actions.items():
        if action == ignore_action:
            continue
        if str(existing or "").strip().lower() == token:
            return action
    for name, existing in binds.slash.items():
        if str(existing or "").strip().lower() == token:
            return "/" + name
    return ""


def set_keybind(action: str, key: str, root: Path | None = None) -> str:
    """Set one action in the profile file, keeping every other entry. Returns "" or an error."""
    name = action.strip()
    if name not in DEFAULT_KEYBINDS:
        return f"error: unknown action {name!r}"
    token = normalise_key(key)
    if token is None:
        return f"error: invalid key {_show(key)} (use one key, for example ctrl+s or f5)"
    base = root or sophon_project_root()
    binds = load_keybinds(base)
    owner = key_owner(binds, token, ignore_action=name)
    if owner:
        return f"error: {token} is already bound to {owner}"
    path = keybinds_path(base)
    try:
        ensure_keybinds_file(base)
        document, problem = _load_document(path)
    except OSError as exc:
        return f"error: {path} cannot be edited safely ({exc.strerror or exc}); fix it by hand"
    if problem is None and document is not None and not isinstance(document, dict):
        problem = "top level is not a mapping"
    if problem is not None:
        return f"error: {path} cannot be edited safely ({problem}); fix it by hand"
    mapping = dict(document or {})
    mapping[name] = token
    data = (_header() + yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True)).encode()
    if len(data) > MAX_KEYBINDS_BYTES:
        return f"error: {path} would exceed 64 KiB; fix it by hand"
    try:
        _atomic_write(path, data)
    except OSError as exc:
        return f"error: could not write {path} ({exc.strerror or exc})"
    return ""


def _atomic_write(path: Path, data: bytes) -> None:
    target = path.resolve()
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".keybinds.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass  # the temp file is already gone; the original error is re-raised below
        raise


def format_keybind_list(binds: KeybindSet | None = None) -> list[str]:
    current = binds or load_keybinds()
    lines = ["keybinds:"]
    for action in DEFAULT_KEYBINDS:
        key = str(current.actions.get(action, "") or "")
        shown = key if key else "(unbound)"
        lines.append(f"  {action}: {shown}")
    return lines


def _clear_action_bindings(screen: object, action: str) -> None:
    bindings = getattr(screen, "_bindings", None)
    keymap = getattr(bindings, "key_to_bindings", None)
    if not isinstance(keymap, dict):
        return
    for chord, items in list(keymap.items()):
        kept = [item for item in items if str(getattr(item, "action", "")) != action]
        if len(kept) == len(list(items)):
            continue
        if kept:
            keymap[chord] = kept
        else:
            del keymap[chord]
            shown = getattr(bindings, "shown_keys", None)
            if isinstance(shown, set):
                shown.discard(chord)


def apply_nav_keybinds(screen: object, binds: KeybindSet | None = None) -> None:
    current = binds or load_keybinds()
    binder = getattr(screen, "bind", None)
    if binder is None:
        return
    for action in NAV_ACTIONS:
        if not callable(getattr(screen, "action_" + action, None)):
            continue
        _clear_action_bindings(screen, action)
        key = str(current.actions.get(action, "") or "").strip()
        if not key:
            continue
        try:
            binder(key, action, show=False, priority=True)
        except Exception:
            continue


def refresh_keybind_surfaces(app: object, binds: KeybindSet | None = None) -> None:
    current = binds or load_keybinds()
    label = panes_footer_label(current)
    stack = list(getattr(app, "screen_stack", []) or [])
    active = getattr(app, "screen", None)
    if active is not None and active not in stack:
        stack.append(active)
    from cli.tui.chat_prompt import ChatPromptInput, sync_prompt_nav
    from cli.tui.footer import NexusFooter

    dash = []
    for action, title in (
        ("open_dashboard", "dashboard"),
        ("open_editor", "editor"),
        ("open_chat", "nexus"),
        ("cycle_mode", "cycle"),
    ):
        shown = format_key_compact(current.actions.get(action, "")) or "-"
        dash.append(f"{shown} {title}")
    dash_text = " · ".join(dash) + " · F4 layout"
    for screen in stack:
        apply_nav_keybinds(screen, current)
        query = getattr(screen, "query", None)
        if not callable(query):
            continue
        for footer in query(NexusFooter):
            setter = getattr(footer, "set_panes_label", None)
            if callable(setter):
                setter(label)
        for prompt in query(ChatPromptInput):
            sync_prompt_nav(prompt, current)
        for notes in query("#dash-notes"):
            update = getattr(notes, "update", None)
            if callable(update):
                update(f"[dim]{dash_text}[/dim]")


def ensure_keybinds_file(root: Path | None = None) -> Path:
    """Create the template if no file exists. Never overwrites. May raise OSError."""
    path = keybinds_path(root)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(_KEYBINDS_TEMPLATE)
    except FileExistsError:
        pass  # created concurrently; keep whatever is there
    return path


def normalise_key(raw: str) -> str | None:
    """Return the normalised key ("" means unbound), or None if it is not exactly one key.

    Spaces around the value are dropped and modifiers are lowercased; the key token keeps
    its case. Commas, inner spaces, brackets and newlines never match.
    """
    value = raw.strip(" ")
    if not value:
        return ""
    parts = value.split("+")
    value = "+".join([part.lower() for part in parts[:-1]] + parts[-1:])
    return value if KEY_PATTERN.fullmatch(value) else None


def _show(value: object) -> str:
    """Quote file content for messages: repr() so control characters are escaped, max 80."""
    text = repr(str(value))
    return text if len(text) <= 80 else text[:77] + "..."


def _label(value: str) -> str:
    return value if _SAFE_LABEL.fullmatch(value) else _show(value)


def _capped(items: list, overflow: object) -> list:
    if len(items) <= MAX_REPORTED:
        return items
    return items[:MAX_REPORTED] + [overflow(len(items) - MAX_REPORTED)]


def _load_document(path: Path) -> tuple[object, str | None]:
    """Bounded read + alias-free parse. Returns (document, None) or (None, problem).

    A missing file is (None, None). Raises nothing for content problems; OSError from
    path checks and reads is reported as a problem too.
    """
    try:
        if not path.is_file():
            return None, None
        with open(path, "rb") as handle:
            data = handle.read(MAX_KEYBINDS_BYTES + 1)
    except OSError as exc:
        return None, f"unreadable: {exc.strerror or exc}"
    if len(data) > MAX_KEYBINDS_BYTES:
        return None, "file larger than 64 KiB"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, "not UTF-8"
    try:
        return yaml.load(text, Loader=_NoAliasLoader), None
    except _AliasRefused:
        return None, "aliases are not allowed"
    except (yaml.YAMLError, RecursionError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return None, f"invalid YAML ({type(exc).__name__})"


def _scalar_key(where: str, value: object, problems: list[str]) -> str | None:
    if not isinstance(value, str):
        problems.append(f"{where}: value is not a single key")
        return None
    if value in _NULL_VALUES:
        return ""
    token = normalise_key(value)
    if token is None:
        problems.append(f"{where}: invalid key {_show(value)}")
    return token


def read_keybinds_file(path: Path) -> RawKeybinds:
    """Read one keybinds file: only its own entries, no defaults, never writes or raises."""
    document, problem = _load_document(path)
    if problem is not None:
        return RawKeybinds({}, {}, (problem,))
    if document is None:
        return RawKeybinds({}, {}, ())
    if not isinstance(document, dict):
        return RawKeybinds({}, {}, ("top level is not a mapping",))
    actions: dict[str, str] = {}
    slash: dict[str, str] = {}
    problems: list[str] = []
    for name, value in document.items():
        if not isinstance(name, str):
            problems.append(f"non-string key {_show(name)}")
            continue
        if name == "slash":
            if isinstance(value, str) and value in _NULL_VALUES:
                continue
            if not isinstance(value, dict):
                problems.append("slash: not a mapping")
                continue
            for command, key in value.items():
                if not isinstance(command, str):
                    problems.append(f"slash: non-string name {_show(command)}")
                    continue
                token = _scalar_key(f"slash {_show(command)}", key, problems)
                if token is not None:
                    slash[command] = token
            continue
        token = _scalar_key(f"action {_label(name)}", value, problems)
        if token is not None:
            actions[name] = token
    capped = _capped(problems, lambda extra: f"and {extra} more")
    return RawKeybinds(actions, slash, tuple(capped))


def _same_dir(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def _load_profile(root: Path | None = None) -> tuple[KeybindSet, tuple[str, ...]]:
    base = root or sophon_project_root()
    problems: list[str] = []
    if _same_dir(base, sophon_project_root()):
        try:
            ensure_keybinds_file(base)
        except OSError as exc:
            problems.append(f"cannot create template: {exc.strerror or exc}")
    raw = read_keybinds_file(keybinds_path(base))
    problems.extend(raw.problems)
    actions = dict(DEFAULT_KEYBINDS)
    actions.update(raw.actions)
    slash = dict(DEFAULT_SLASH_BINDS)
    for name, key in raw.slash.items():
        if not name or len(name) > MAX_SLASH_NAME or PROFILE_SLASH_BAD.search(name):
            problems.append(f"slash {_show(name)}: invalid command name")
            continue
        command = name.strip(" ").lstrip("/")
        if command:
            slash[command] = key
    capped = _capped(problems, lambda extra: f"and {extra} more")
    return KeybindSet(actions=actions, slash=slash), tuple(capped)


def load_keybinds(root: Path | None = None) -> KeybindSet:
    """The profile (defaults + the profile file). Only the profile root gets a template."""
    return _load_profile(root)[0]


def profile_keybind_problems(root: Path | None = None) -> tuple[str, ...]:
    """Entries of the profile file that were rejected, for /keybind."""
    return _load_profile(root)[1]


def load_effective_keybinds(workspace: Path | None) -> KeybindLoad:
    """Profile overlaid with the allow-listed, validated entries of a workspace keybinds file."""
    profile = load_keybinds()
    if workspace is None:
        return KeybindLoad(profile, (), None)
    try:
        if _same_dir(workspace, sophon_project_root()):
            return KeybindLoad(profile, (), None)
        ws_file = keybinds_path(workspace)
        if not ws_file.is_file():
            return KeybindLoad(profile, (), None)
    except OSError as exc:
        return KeybindLoad(profile, (("file", f"unreadable: {exc.strerror or exc}"),), None)
    raw = read_keybinds_file(ws_file)
    ignored: list[tuple[str, str]] = [("file", problem) for problem in raw.problems]
    actions = dict(profile.actions)
    used: set[str] = set()
    for name, key in raw.actions.items():
        entry = f"action {_label(name)}"
        reason = ""
        if name not in WORKSPACE_ACTIONS:
            reason = "protected action" if name in DEFAULT_KEYBINDS else "unknown action"
        elif key.lower() in WORKSPACE_RESERVED:
            reason = "reserved key"
        elif key and not WORKSPACE_KEY_OK.fullmatch(key) and key != profile.actions.get(name):
            reason = "key needs a modifier or F-key"
        elif key and (key_owner(profile, key, ignore_action=name) or key.lower() in used):
            reason = "key already bound"
        if reason:
            ignored.append((entry, reason))
            continue
        actions[name] = key
        if key:
            used.add(key.lower())
    for command in raw.slash:
        ignored.append((f"slash {_show('/' + command)}", "slash commands are profile-only"))
    capped = _capped(ignored, lambda extra: ("...", f"and {extra} more"))
    binds = KeybindSet(actions=actions, slash=dict(profile.slash))
    return KeybindLoad(binds, tuple(capped), ws_file)


def apply_keybinds(screen: object, binds: KeybindSet, *, actions: frozenset[str] | None = None) -> None:
    binder = getattr(screen, "bind", None)
    if binder is None:
        return
    notify = getattr(screen, "notify", None)
    for action, key in binds.actions.items():
        if actions is not None and action not in actions:
            continue
        if not key:
            continue
        if not callable(getattr(screen, "action_" + action, None)):
            continue
        try:
            binder(key, action, show=False)
        except Exception:
            if callable(notify):
                notify(f"Invalid keybind {action}: {key}")
    for name, key in binds.slash.items():
        if not key:
            continue
        try:
            binder(key, "run_slash('" + name + "')", show=False)
        except Exception:
            if callable(notify):
                notify(f"Invalid slash keybind /{name}: {key}")
