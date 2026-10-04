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


def save_keybinds(binds: KeybindSet, root: Path | None = None) -> Path:
    path = keybinds_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    for action in DEFAULT_KEYBINDS:
        payload[action] = binds.actions.get(action, "")
    for action, key in binds.actions.items():
        if action not in payload:
            payload[action] = key
    payload["slash"] = {name: binds.slash.get(name, "") for name in DEFAULT_SLASH_BINDS}
    for name, key in binds.slash.items():
        slash = payload["slash"]
        if isinstance(slash, dict) and name not in slash:
            slash[name] = key
    path.write_text(_header() + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


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
    name = action.strip()
    if name not in DEFAULT_KEYBINDS:
        return f"error: unknown action {name!r}"
    binds = load_keybinds(root)
    token = key.strip()
    owner = key_owner(binds, token, ignore_action=name)
    if owner:
        return f"error: {token} is already bound to {owner}"
    binds.actions[name] = token
    save_keybinds(binds, root)
    return ""


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
