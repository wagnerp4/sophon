from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from utils.device.platform import (
    is_wsl,
    linux_path_to_windows,
    resolve_windows_console_script,
    store_pwsh_windows,
    windows_local_appdata_linux,
)

PROFILE_NAME = "sophon"
LEGACY_PROFILE_GUID = "{8f4e2c91-6b3a-4d1e-9c0f-mithrilchat01}"
PROFILE_GUID = "{8f4e2c91-6b3a-4d1e-9c0f-a1b2c3d40101}"


def _project_root() -> Path:
    try:
        from utils.device.env_bootstrap import sophon_project_root

        return sophon_project_root()
    except Exception:
        return Path.cwd().resolve()


def _runtime_root_windows() -> str:
    from cli.host.windows_deploy import windows_runtime_root

    return windows_runtime_root()


def profile_icon_path() -> Path | None:
    from utils.device.env_bootstrap import sophon_assets_dir

    assets = sophon_assets_dir()
    names = ("terminal-icon.png", "terminal-icon.ico", "terminal-icon.svg")
    for name in names:
        candidate = (assets / name).resolve()
        if candidate.is_file():
            return candidate
    return None


def _profile_icon_windows() -> str | None:
    from cli.host.windows_deploy import linux_runtime_root

    names = ("terminal-icon.png", "terminal-icon.ico", "terminal-icon.svg")
    try:
        assets = linux_runtime_root() / "data" / "assets"
    except RuntimeError:
        assets = None
    if assets is not None:
        for name in names:
            candidate = assets / name
            if candidate.is_file():
                return linux_path_to_windows(candidate) or str(candidate)
    icon = profile_icon_path()
    if icon is None:
        return None
    return linux_path_to_windows(icon) or str(icon)


def _chat_launcher_command() -> str:
    runtime = _runtime_root_windows()
    start_ps1 = f"{runtime}\\scripts\\start-sophon-chat.ps1"
    pwsh = store_pwsh_windows()
    return (
        f"{pwsh} -NoLogo -NoProfile -ExecutionPolicy Bypass "
        f"-File \"{start_ps1}\" -Foreground"
    )


def _windows_settings_paths() -> list[Path]:
    locals_: list[Path] = []
    env_local = os.environ.get("LOCALAPPDATA", "").strip()
    if env_local:
        locals_.append(Path(env_local))
    wsl_local = windows_local_appdata_linux()
    if wsl_local is not None:
        locals_.append(wsl_local)

    paths: list[Path] = []
    seen: set[str] = set()
    for local in locals_:
        key = str(local)
        if key in seen:
            continue
        seen.add(key)
        paths.append(local / "Microsoft" / "Windows Terminal" / "settings.json")
        paths.append(
            local
            / "Packages"
            / "Microsoft.WindowsTerminal_8wekyb3d8bbwe"
            / "LocalState"
            / "settings.json"
        )
    return paths


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
        known = _windows_settings_paths()
        if not known:
            return None
        settings_path = known[0]
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.is_file():
        data = _read_json(settings_path)
    else:
        data = {
            "$schema": "https://aka.ms/terminal-profiles-schema",
            "profiles": {"defaults": {}, "list": []},
        }

    profiles = data.setdefault("profiles", {}).setdefault("list", [])
    root = _runtime_root_windows()
    icon = _profile_icon_windows()
    profile: dict[str, Any] = {
        "guid": PROFILE_GUID,
        "name": PROFILE_NAME,
        "commandline": _chat_launcher_command(),
        "startingDirectory": root,
        "hidden": False,
        "tabTitle": "sophon",
        "colorScheme": "Campbell Powershell",
    }
    if icon is not None:
        profile["icon"] = icon

    replaced = False
    for index, item in enumerate(profiles):
        if (
            item.get("guid") in (PROFILE_GUID, LEGACY_PROFILE_GUID)
            or item.get("name") in (PROFILE_NAME, "orodruin", "mithril")
        ):
            profiles[index] = {**item, **profile}
            replaced = True
            break
    if not replaced:
        profiles.append(profile)

    settings_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return settings_path


def install_macos_terminal_profile() -> Path:
    from utils.device.env_bootstrap import sophon_chat_logs_dir

    root = _project_root()
    target = sophon_chat_logs_dir() / "macos-terminal-profile.txt"
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
    from utils.device.env_bootstrap import sophon_chat_logs_dir

    root = _project_root()
    target = sophon_chat_logs_dir() / "linux-terminal-profile.desktop"
    target.parent.mkdir(parents=True, exist_ok=True)
    win_script = resolve_windows_console_script("sophon-chat-tui", root)
    if win_script is not None:
        launcher = f"\"{win_script}\" --no-spawn-window"
    else:
        scripts_dir = Path(sys.executable).resolve().parent
        launcher = None
        for script_name in ("sophon-chat-tui.exe", "sophon-chat-tui"):
            script_path = scripts_dir / script_name
            if script_path.is_file():
                launcher = f"\"{script_path}\" --no-spawn-window"
                break
        if launcher is None:
            launcher = f"\"{sys.executable}\" -m sophon.cli.backends.textual --no-spawn-window"
    icon = profile_icon_path()
    icon_line = f"Icon={icon}\n" if icon is not None else ""
    content = "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            "Name=sophon chat",
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
    if sys.platform == "win32" or is_wsl():
        path = install_windows_terminal_profile()
        if path is not None:
            messages.append(f"Windows Terminal profile installed: {path}")
            messages.append(
                f"Profile name: {PROFILE_NAME!r} (Store pwsh + start-sophon-chat.ps1)"
            )
        else:
            messages.append("Windows Terminal settings.json was not found.")
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
