from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ORODRUIN_CONSOLE_STEMS = (
    "orodruin-cli",
    "orodruin-infer",
    "orodruin-chat-cli",
    "orodruin-chat-tui",
    "orodruin-download-hf",
    "orodruin-download-hf-debug",
    "orodruin-hf-download",
    "orodruin-system-check",
    "orodruin-benchmark",
)

_CHILD_ENV_KEYS = (
    "ORODRUIN_TUI_LOG",
    "ORODRUIN_TUI_CHILD",
    "ORODRUIN_INTERFACE",
    "PYTHONPATH",
    "PYTHONUTF8",
    "PYTHONIOENCODING",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "ORODRUIN_ENV_FILE",
    "ORODRUIN_OBSIDIAN_TOOLS",
    "ORODRUIN_OBSIDIAN_API_URL",
    "ORODRUIN_OBSIDIAN_API_KEY",
    "OBSIDIAN_API_KEY",
    "OBSIDIAN_API_TOKEN",
    "ORODRUIN_SHELL_TOOLS",
    "ORODRUIN_SHELL_TIMEOUT_S",
    "ORODRUIN_TTS_TOOL",
    "ORODRUIN_TTS_BACKEND",
    "ORODRUIN_TTS_SPEAKER",
    "ORODRUIN_TTS_SPEED",
    "ORODRUIN_SST",
    "ORODRUIN_SST_TOOL",
    "ORODRUIN_SST_BACKEND",
    "ORODRUIN_SST_MODEL",
    "ORODRUIN_SST_LANGUAGE",
    "ORODRUIN_SST_DEVICE",
    "ORODRUIN_SST_MAX_NEW_TOKENS",
    "ORODRUIN_SST_LISTEN_S",
    "ORODRUIN_SST_VAD_RMS",
    "ORODRUIN_SST_MIC",
    "ORODRUIN_LISTEN_SECONDS",
)


def is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def is_windows_mount_path(path: str | Path) -> bool:
    normalized = str(Path(path).expanduser())
    return normalized.startswith("/mnt/") or normalized.startswith("\\\\")


def should_disable_mmap(model_path: str | Path) -> bool:
    if sys.platform == "win32":
        return True
    return False


def venv_root() -> Path | None:
    prefix = Path(sys.prefix).resolve()
    if (prefix / "pyvenv.cfg").is_file():
        return prefix
    return None


def venv_bin_dir() -> Path:
    root = venv_root()
    if root is not None:
        if sys.platform == "win32":
            scripts = root / "Scripts"
            if scripts.is_dir():
                return scripts
        bin_dir = root / "bin"
        if bin_dir.is_dir():
            return bin_dir
    return Path(sys.executable).resolve().parent


def is_windows_pe_executable(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    if path.suffix.lower() != ".exe":
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(2) == b"MZ"
    except OSError:
        return False


def resolve_windows_console_script(stem: str, project_root: Path | None = None) -> Path | None:
    if project_root is None:
        from utils.device.env_bootstrap import orodruin_project_root

        project_root = orodruin_project_root()
    candidate = Path(project_root) / ".venv" / "Scripts" / f"{stem}.exe"
    return candidate if is_windows_pe_executable(candidate) else None


def linux_path_to_windows(path: str | Path) -> str | None:
    if sys.platform == "win32":
        return str(path)
    if not is_wsl():
        return None
    wslpath = shutil.which("wslpath")
    if not wslpath:
        return None
    try:
        proc = subprocess.run(
            [wslpath, "-w", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    out = proc.stdout.strip()
    return out or None


def wsl_windows_venv_message() -> str:
    from utils.device.env_bootstrap import orodruin_project_root

    root = orodruin_project_root()
    win_root = linux_path_to_windows(root) or str(root)
    return "\n".join(
        [
            "orodruin desktop TUI requires the Windows virtual environment (.venv/Scripts/orodruin-cli.exe).",
            "",
            "Close any running orodruin-chat-tui / Windows Terminal orodruin windows first.",
            "If uv sync reports Access is denied, remove .venv from PowerShell (not WSL).",
            "",
            "From PowerShell on Windows:",
            f"  cd {win_root}",
            "  Remove-Item -Recurse -Force .venv",
            "  $env:UV_LINK_MODE = 'copy'",
            "  uv sync --extra tui --extra finetune",
            "",
            "Then from WSL or PowerShell:",
            "  ./.venv/Scripts/orodruin-cli.exe chat --preset llama2_7b_chat",
            "",
            "From WSL (delegates to the Windows .exe when present):",
            "  orodruin-cli chat --windows",
            "",
            "Stay on WSL/Linux Python instead:",
            "  orodruin-cli chat --linux",
        ]
    )


_WSL_PLATFORM_FLAGS = frozenset(
    {
        "--windows",
        "--win",
        "--win-exe",
        "--linux",
        "--wsl",
        "--no-windows",
    }
)


def strip_wsl_platform_flags(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    for arg in argv:
        if arg in _WSL_PLATFORM_FLAGS:
            continue
        cleaned.append(arg)
    return cleaned


def wsl_windows_cli_preference(*, windows: bool = False, linux: bool = False) -> str:
    if linux:
        return "linux"
    if windows:
        return "windows"
    env = os.environ.get("ORODRUIN_WINDOWS_CLI", "").strip().lower()
    if env in ("1", "true", "yes", "windows", "win"):
        return "windows"
    if env in ("0", "false", "no", "linux", "wsl"):
        return "linux"
    return "auto"


def windows_console_script_available(stem: str = "orodruin-cli") -> bool:
    return resolve_windows_console_script(stem) is not None


def delegate_wsl_to_windows_cli(
    stem: str,
    user_argv: list[str],
    *,
    preference: str = "auto",
) -> None:
    if not is_wsl() or sys.platform == "win32":
        return
    if preference == "linux":
        return
    want_windows = preference == "windows" or preference == "auto"
    if not want_windows:
        return
    if not windows_console_script_available(stem):
        if preference == "windows":
            raise SystemExit(wsl_windows_venv_message())
        return
    cleaned = strip_wsl_platform_flags(user_argv)
    code = wsl_reexec_windows_cli(stem, cleaned)
    if code is None:
        if preference == "windows":
            raise SystemExit(wsl_windows_venv_message())
        return
    raise SystemExit(int(code))


def wsl_reexec_windows_cli(stem: str, user_argv: list[str]) -> int | None:
    if not is_wsl() or sys.platform == "win32":
        return None
    exe = resolve_windows_console_script(stem)
    if exe is None:
        return None
    win_exe = linux_path_to_windows(exe)
    cmd = shutil.which("cmd.exe")
    if not win_exe or not cmd:
        return None
    forward = [win_exe, *strip_wsl_platform_flags(user_argv)]
    line = subprocess.list2cmdline(forward)
    return subprocess.call([cmd, "/c", line])


def resolve_installed_script(stem: str) -> Path | None:
    scripts = venv_bin_dir()
    if sys.platform == "win32":
        candidates = (scripts / f"{stem}.exe", scripts / stem)
    else:
        candidates = (scripts / stem, scripts / f"{stem}.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def module_fallback_argv(module: str, args: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *args]


def chat_tui_argv(args: list[str]) -> list[str]:
    if sys.platform == "win32":
        win_script = resolve_windows_console_script("orodruin-chat-tui")
        if win_script is not None:
            return [str(win_script), *args]
    script = resolve_installed_script("orodruin-chat-tui")
    if script is not None:
        return [str(script), *args]
    return module_fallback_argv("orodruin.cli.backends.textual", args)


def desktop_terminal_available() -> bool:
    if sys.platform == "win32":
        return True
    if sys.platform == "darwin":
        return True
    if is_wsl() and resolve_windows_console_script("orodruin-cli") is not None:
        return True
    return False


def child_env_keys() -> tuple[str, ...]:
    return _CHILD_ENV_KEYS


def format_cli_hint(stem: str) -> str:
    if sys.platform == "win32" or is_wsl():
        win_script = resolve_windows_console_script(stem)
        if win_script is not None:
            return str(win_script)
    script = resolve_installed_script(stem)
    if script is not None:
        return str(script)
    if sys.platform == "win32" or is_wsl():
        root = venv_root()
        if root is not None:
            return str(root / "Scripts" / f"{stem}.exe")
    return str(venv_bin_dir() / stem)


def ensure_venv_scripts_shims() -> list[Path]:
    if sys.platform == "win32":
        return []
    root = venv_root()
    if root is None:
        return []
    bin_dir = root / "bin"
    scripts_dir = root / "Scripts"
    if not bin_dir.is_dir():
        return []
    created: list[Path] = []
    scripts_dir.mkdir(exist_ok=True)
    for stem in ORODRUIN_CONSOLE_STEMS:
        source = bin_dir / stem
        if not source.is_file():
            continue
        link_target = Path("..") / "bin" / stem
        for name in (f"{stem}.exe", stem):
            destination = scripts_dir / name
            if destination.exists() or destination.is_symlink():
                continue
            try:
                destination.symlink_to(link_target)
                created.append(destination)
            except OSError:
                continue
    return created


def sync_shims_main() -> None:
    created = ensure_venv_scripts_shims()
    if not created:
        print("orodruin: Scripts shims already present (or bin/ entrypoints missing).", flush=True)
        return
    print("orodruin: created Scripts shims for cross-platform paths:", flush=True)
    for path in created:
        print(f"  {path}", flush=True)
