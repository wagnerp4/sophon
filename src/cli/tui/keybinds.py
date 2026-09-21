from __future__ import annotations

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
}

DEFAULT_SLASH_BINDS: dict[str, str] = {
    "models": "ctrl+shift+m",
    "permissions": "f2",
}

CHAT_KEYBIND_ACTIONS = frozenset({"copy_chat", "listen", "open_models"})

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
slash:
  models: ctrl+shift+m
  permissions: f2
"""


@dataclass
class KeybindSet:
    actions: dict[str, str] = field(default_factory=dict)
    slash: dict[str, str] = field(default_factory=dict)


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


def ensure_keybinds_file(root: Path | None = None) -> Path:
    path = keybinds_path(root)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_KEYBINDS_TEMPLATE, encoding="utf-8")
    return path


def load_keybinds(root: Path | None = None) -> KeybindSet:
    path = ensure_keybinds_file(root)
    actions = dict(DEFAULT_KEYBINDS)
    slash = dict(DEFAULT_SLASH_BINDS)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return KeybindSet(actions=actions, slash=slash)
    if not isinstance(loaded, dict):
        return KeybindSet(actions=actions, slash=slash)
    nested = loaded.get("slash")
    if isinstance(nested, dict):
        for name, key in nested.items():
            cmd = str(name).strip().lstrip("/")
            if not cmd:
                continue
            if key is None:
                slash[cmd] = ""
                continue
            slash[cmd] = str(key).strip()
    for action, key in loaded.items():
        name = str(action).strip()
        if name == "slash":
            continue
        if isinstance(key, dict):
            continue
        if key is None:
            actions[name] = ""
            continue
        actions[name] = str(key).strip()
    return KeybindSet(actions=actions, slash=slash)


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
