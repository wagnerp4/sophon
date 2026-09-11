from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROFILE_NAME = "orodruin"
LEGACY_PROFILE_GUID = "{8f4e2c91-6b3a-4d1e-9c0f-mithrilchat01}"
PROFILE_GUID = "{8f4e2c91-6b3a-4d1e-9c0f-a1b2c3d40101}"


def _project_root() -> Path:
    try:
        from utils.device.env_bootstrap import orodruin_project_root

        return orodruin_project_root()
    except Exception:
        return Path.cwd().resolve()


def profile_icon_path() -> Path | None:
    from utils.device.env_bootstrap import orodruin_assets_dir

    assets = orodruin_assets_dir()
    names = ("terminal-icon.png", "terminal-icon.ico", "terminal-icon.svg")
    for name in names:
        candidate = (assets / name).resolve()
        if candidate.is_file():
            return candidate
    return None


def _chat_launcher_command() -> str:
    from utils.device.platform import resolve_windows_console_script

    win_script = resolve_windows_console_script("orodruin-chat-tui", _project_root())
    if win_script is not None:
        return f"\"{win_script}\" --no-spawn-window"
    scripts_dir = Path(sys.executable).resolve().parent
    for script_name in ("orodruin-chat-tui.exe", "orodruin-chat-tui"):
        script_path = scripts_dir / script_name
        if script_path.is_file():
            return f"\"{script_path}\" --no-spawn-window"
    return f"\"{sys.executable}\" -m orodruin.cli.backends.textual --no-spawn-window"


def _windows_settings_paths() -> list[Path]:
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    if not local:
        return []
    return [
        local / "Microsoft" / "Windows Terminal" / "settings.json",
        local / "Packages" / "Microsoft.WindowsTerminal_8wekyb3d8bbwe" / "LocalState" / "settings.json",
    ]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _windows_profile_installed(settings_path: Path) -> bool:
    if not settings_path.is_file():
        return False
    try:
        data = _read_json(settings_path)
    except Exception:
        return False
    profiles = data.get("profiles", {}).get("list", [])
    return any(item.get("name") == PROFILE_NAME for item in profiles)


def windows_profile_installed() -> bool:
    return any(_windows_profile_installed(path) for path in _windows_settings_paths())


def install_windows_terminal_profile() -> Path | None:
    settings_path = next((path for path in _windows_settings_paths() if path.parent.exists()), None)
    if settings_path is None:
        settings_path = _windows_settings_paths()[0]
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.is_file():
        data = _read_json(settings_path)
    else:
        data = {
            "$schema": "https://aka.ms/terminal-profiles-schema",
            "profiles": {"defaults": {}, "list": []},
        }

    profiles = data.setdefault("profiles", {}).setdefault("list", [])
    root = _project_root()
    icon = profile_icon_path()
    profile: dict[str, Any] = {
        "guid": PROFILE_GUID,
        "name": PROFILE_NAME,
        "commandline": _chat_launcher_command(),
        "startingDirectory": str(root),
        "hidden": False,
        "tabTitle": "orodruin",
        "colorScheme": "Campbell Powershell",
    }
    if icon is not None:
        profile["icon"] = str(icon)

    replaced = False
    for index, item in enumerate(profiles):
        if (
            item.get("guid") in (PROFILE_GUID, LEGACY_PROFILE_GUID)
            or item.get("name") in (PROFILE_NAME, "mithril")
        ):
            profiles[index] = {**item, **profile}
            replaced = True
            break
    if not replaced:
        profiles.append(profile)

    settings_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return settings_path


def install_macos_terminal_profile() -> Path:
    from utils.device.env_bootstrap import orodruin_chat_logs_dir

    root = _project_root()
    target = orodruin_chat_logs_dir() / "macos-terminal-profile.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    launcher = _chat_launcher_command().replace('"', "")
    lines = [
        "macOS Terminal profile (manual setup)",
        "",
        "1. Open Terminal > Settings > Profiles > +",
        f"2. Name: {PROFILE_NAME}",
        f"3. Working directory: {root}",
        f"4. Run command: {launcher}",
        "5. Optional: drag data/assets/terminal-icon.svg into the profile icon field",
        "",
        "Native terminals on macOS: Terminal.app, iTerm2, or Ghostty/Zentty.",
        "Set the same command in whichever host you prefer.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def install_linux_terminal_profile() -> Path:
    from utils.device.env_bootstrap import orodruin_chat_logs_dir

    root = _project_root()
    target = orodruin_chat_logs_dir() / "linux-terminal-profile.desktop"
    target.parent.mkdir(parents=True, exist_ok=True)
    launcher = _chat_launcher_command()
    icon = profile_icon_path()
    icon_line = f"Icon={icon}\n" if icon is not None else ""
    content = "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            "Name=orodruin chat",
            "Comment=Local HF chat TUI",
            f"Path={root}",
            f"Exec=/bin/sh -lc 'cd {shlex_quote(str(root))} && {launcher}'",
            "Terminal=true",
            "Categories=Development;",
            icon_line.rstrip(),
            "",
        ]
    )
    target.write_text(content, encoding="utf-8")
    return target


def shlex_quote(value: str) -> str:
    if sys.platform == "win32":
        if value == "":
            return '""'
        if any(ch in value for ch in ' \t"&()[]{}^=;!\'`'):
            return '"' + value.replace('"', '\\"') + '"'
        return value
    import shlex

    return shlex.quote(value)


def install_terminal_profiles() -> list[str]:
    messages: list[str] = []
    if sys.platform == "win32":
        path = install_windows_terminal_profile()
        if path is not None:
            messages.append(f"Windows Terminal profile installed: {path}")
            messages.append(
                f"Profile name: {PROFILE_NAME!r} (custom icon if data/assets/terminal-icon.* exists)"
            )
    elif sys.platform == "darwin":
        path = install_macos_terminal_profile()
        messages.append(f"macOS profile instructions written: {path}")
    else:
        path = install_linux_terminal_profile()
        messages.append(f"Linux launcher stub written: {path}")
        messages.append("Copy to ~/.local/share/applications/ if you want a menu entry.")
    return messages


def profile_name() -> str:
    return PROFILE_NAME
