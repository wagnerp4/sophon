from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _uv_toad_candidates() -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "toad.exe",
        home / ".local" / "bin" / "toad",
    ]
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidates.append(
            Path(appdata) / "uv" / "tools" / "batrachian-toad" / "Scripts" / "toad.exe"
        )
    return candidates


def _local_toad_repo_candidates() -> list[Path]:
    explicit = os.environ.get("ORODRUIN_TOAD_HOME", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    try:
        from utils.device.env_bootstrap import orodruin_project_root

        root = orodruin_project_root()
        candidates.extend(
            [
                root.parent / "Repos" / "Agentic" / "toad",
                root / ".." / "Repos" / "Agentic" / "toad",
            ]
        )
    except Exception:
        pass
    return candidates


def resolve_toad_command() -> list[str] | None:
    explicit = os.environ.get("ORODRUIN_TOAD_BIN", "").strip() or os.environ.get("TOAD_BIN", "").strip()
    if explicit:
        return [explicit]

    found = shutil.which("toad")
    if found:
        return [found]

    for candidate in _uv_toad_candidates():
        if candidate.is_file():
            return [str(candidate.resolve())]

    uv_bin = shutil.which("uv")
    for repo in _local_toad_repo_candidates():
        repo = repo.resolve()
        if not repo.is_dir():
            continue
        for rel in (
            Path(".venv") / "Scripts" / "toad.exe",
            Path(".venv") / "bin" / "toad",
        ):
            script = repo / rel
            if script.is_file():
                return [str(script.resolve())]
        if uv_bin and (repo / "pyproject.toml").is_file():
            return [uv_bin, "run", "--directory", str(repo), "toad"]

    return None


def _toad_windows_native_unstable() -> bool:
    if sys.platform != "win32":
        return False
    allow = os.environ.get("ORODRUIN_TOAD_ALLOW_WINDOWS", "").strip().lower()
    return allow not in ("1", "true", "yes")


def _toad_windows_blocked_message() -> str:
    lines = [
        "Toad is not supported on native Windows (upstream targets Linux/macOS).",
        "Launching it here typically exits with code 0xC0000005 (access violation).",
        "",
        "Use orodruin Textual instead:",
        "  orodruin-cli chat --tui --interface textual --preset llama2_7b_chat",
        "",
        "Or run Toad inside WSL/Linux/macOS, then bridge orodruin with:",
        "  !orodruin-chat-tui --no-spawn-window --preset llama2_7b_chat",
        "",
        "To attempt native Windows launch anyway (UTF-8 console, no orodruin WT profile):",
        "  $env:ORODRUIN_TOAD_ALLOW_WINDOWS = \"1\"",
        "  orodruin-cli chat --interface toad --preset llama2_7b_chat",
    ]
    return "\n".join(lines)


def _toad_missing_message() -> str:
    lines = [
        "toad not found.",
        "",
        "Install (requires Python 3.14):",
        "  uv python install 3.14",
        "  uv tool install -U batrachian-toad --python 3.14",
        "",
        "Then ensure uv tools are on PATH (PowerShell, current session):",
        '  $env:PATH = "$env:USERPROFILE\\.local\\bin;$env:PATH"',
        "",
        "Or point orodruin at a specific binary:",
        "  $env:ORODRUIN_TOAD_BIN = \"$env:USERPROFILE\\.local\\bin\\toad.exe\"",
        "",
        "Local clone option:",
        "  $env:ORODRUIN_TOAD_HOME = \"C:\\Software\\Python\\NLP\\Repos\\Agentic\\toad\"",
        "  cd $env:ORODRUIN_TOAD_HOME; uv sync",
        "",
        "Note: upstream Toad targets Linux/macOS first. On Windows, WSL is the most",
        "stable host if native launch misbehaves.",
        "",
        "Use orodruin Textual instead:",
        "  orodruin-cli chat --tui --interface textual --preset llama2_7b_chat",
    ]
    return "\n".join(lines)


def launch(user_argv: list[str]) -> int:
    command = resolve_toad_command()
    if command is None:
        raise SystemExit(_toad_missing_message())

    if _toad_windows_native_unstable():
        raise SystemExit(_toad_windows_blocked_message())

    try:
        from utils.device.env_bootstrap import orodruin_project_root

        root = orodruin_project_root()
    except Exception:
        root = None

    if root is not None:
        guide = root / "integrations" / "toad" / "README.md"
        if guide.is_file():
            print(f"[orodruin] Toad integration guide: {guide}", flush=True)

    print(f"[orodruin] Using Toad: {' '.join(command)}", flush=True)
    print(
        "[orodruin] Inside Toad shell run: !orodruin-chat-tui --no-spawn-window --preset llama2_7b_chat",
        flush=True,
    )

    from cli.host.spawn import child_runtime_env, project_root, spawn_desktop_terminal

    env = child_runtime_env(os.environ.copy())
    env["ORODRUIN_INTERFACE"] = "toad"
    cwd = str(project_root())
    spawn_desktop_terminal(
        command,
        env,
        cwd,
        window_title="toad",
        use_orodruin_profile=False,
        utf8_console=True,
    )
    print("[orodruin] Toad opened in a desktop terminal (UTF-8 console).", flush=True)
    print(
        "[orodruin] If the window closes instantly, Toad may not be stable on native Windows.",
        flush=True,
    )
    print(
        "[orodruin] Use WSL/Linux/macOS, or orodruin Textual: orodruin-cli chat --tui --interface textual",
        flush=True,
    )
    return 0
