from __future__ import annotations

from pathlib import Path

import yaml

from utils.device.env_bootstrap import orodruin_project_root

PROFILE_DIRNAME = ".orodruin"
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
    "toggle_project": "ctrl+b",
    "toggle_aux": "ctrl+j",
    "accept_completion": "right",
}

_KEYBINDS_TEMPLATE = """# orodruin editor keybinds (project profile)
# Restart is not required if you save this file from the editor.
# Textual key names: ctrl+s, ctrl+shift+o, right
# Empty quotes unbind an action.

save_file: ctrl+s
save_as: ctrl+shift+s
new_file: ctrl+n
new_folder: ""
open_file: ctrl+o
open_folder: ctrl+shift+o
find_in_file: ctrl+f
refresh_tree: ctrl+r
toggle_project: ctrl+b
toggle_aux: ctrl+j
accept_completion: right
"""


def profile_dir(root: Path | None = None) -> Path:
    return (root or orodruin_project_root()) / PROFILE_DIRNAME


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


def load_keybinds(root: Path | None = None) -> dict[str, str]:
    path = ensure_keybinds_file(root)
    merged = dict(DEFAULT_KEYBINDS)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return merged
    if not isinstance(loaded, dict):
        return merged
    for action, key in loaded.items():
        name = str(action).strip()
        if name not in DEFAULT_KEYBINDS:
            continue
        if key is None:
            merged[name] = ""
            continue
        merged[name] = str(key).strip()
    return merged
