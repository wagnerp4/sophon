from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_SOPHON_WINDOWS_ROOT = r"C:\Software\Python\NLP\Personal\sophon"

SOPHON_CONSOLE_STEMS = (
    "sophon-cli",
    "sophon-infer",
    "sophon-chat-cli",
    "sophon-chat-tui",
    "sophon-download-hf",
    "sophon-download-hf-debug",
    "sophon-hf-download",
    "sophon-system-check",
    "sophon-benchmark",
    "sophon-rag-index",
    "sophon-sync-shims",
    "sophon-finetune",
)

_CHILD_ENV_KEYS = (
    "SOPHON_TUI_LOG",
    "SOPHON_TUI_CHILD",
    "SOPHON_INTERFACE",
    "PYTHONPATH",
    "PYTHONUTF8",
    "PYTHONIOENCODING",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "SOPHON_ENV_FILE",
    "SOPHON_OBSIDIAN_TOOLS",
    "SOPHON_OBSIDIAN_API_URL",
    "SOPHON_OBSIDIAN_API_KEY",
    "OBSIDIAN_API_KEY",
    "OBSIDIAN_API_TOKEN",
    "SOPHON_SHELL_TOOLS",
    "SOPHON_SHELL_TIMEOUT_S",
    "SOPHON_EDITOR_TOOLS",
    "SOPHON_MEMORY_DB",
    "SOPHON_MEMORY_SESSION",
    "SOPHON_MEMORY_USER",
    "SOPHON_MEMORY_RECALL_TURNS",
    "SOPHON_MEMORY_DIR",
    "SOPHON_MEMORY_BUDGET_CHARS",
    "SOPHON_MEMORY_TOOLS",
    "SOPHON_SKILLS",
    "SOPHON_SKILLS_DIRS",
    "SOPHON_SKILLS_CATALOG_CHARS",
    "SOPHON_SKILL_TOOLS",
    "SOPHON_TTS_TOOL",
    "SOPHON_TTS_BACKEND",
    "SOPHON_TTS_SPEAKER",
    "SOPHON_TTS_SPEED",
    "SOPHON_SST",
    "SOPHON_SST_TOOL",
    "SOPHON_SST_BACKEND",
    "SOPHON_SST_MODEL",
    "SOPHON_SST_LANGUAGE",
    "SOPHON_SST_DEVICE",
    "SOPHON_SST_MAX_NEW_TOKENS",
    "SOPHON_SST_MAX_S",
    "SOPHON_SST_LISTEN_S",
    "SOPHON_SST_VAD_RMS",
    "SOPHON_SST_MIC",
    "SOPHON_LISTEN_SECONDS",
    "SOPHON_GOOGLE_CLIENT_SECRETS",
    "SOPHON_GOOGLE_TOOLS",
    "SOPHON_BOOKMARKS_PATH",
    "SOPHON_GOOGLE_CSE_KEY",
    "SOPHON_GOOGLE_CSE_CX",
    "SOPHON_WEB_SEARCH_TOOLS",
    "SOPHON_SEARXNG_URL",
    "SOPHON_SEARXNG_ENGINES",
    "SOPHON_SEARXNG_CATEGORIES",
    "SOPHON_CHAT_BACKEND",
    "SOPHON_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "SOPHON_OPENAI_BASE_URL",
    "OPENAI_BASE_URL",
    "SOPHON_OPENAI_TIMEOUT_S",
    "SOPHON_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
    "SOPHON_ANTHROPIC_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "SOPHON_ANTHROPIC_TIMEOUT_S",
    "SOPHON_GEMINI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "SOPHON_GEMINI_BASE_URL",
    "GEMINI_BASE_URL",
    "SOPHON_GEMINI_TIMEOUT_S",
    "SOPHON_OVERLEAF_TOOLS",
    "SOPHON_OVERLEAF_GIT_TOKEN",
    "OVERLEAF_GIT_TOKEN",
    "SOPHON_OVERLEAF_PROJECT_ID",
    "SOPHON_OVERLEAF_PROJECT_IDS",
    "SOPHON_WINDOWS_ROOT",
    "SOPHON_WINDOWS_CLI",
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


def _windows_runtime_roots(project_root: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path | None) -> None:
        if path is None:
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    if project_root is not None:
        _add(Path(project_root))
    else:
        try:
            from utils.device.env_bootstrap import sophon_project_root

            _add(sophon_project_root())
        except Exception:
            pass
    _add(sophon_windows_root_linux())
    return roots


def resolve_windows_console_script(stem: str, project_root: Path | None = None) -> Path | None:
    for root in _windows_runtime_roots(project_root):
        candidate = root / ".venv" / "Scripts" / f"{stem}.exe"
        if is_windows_pe_executable(candidate):
            return candidate
    return None


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


def _wsl_exe_wslpath_u(windows_path: str) -> str | None:
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return None
    try:
        proc = subprocess.run(
            [wsl, "-e", "wslpath", "-u", windows_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    out = proc.stdout.strip().splitlines()
    if not out:
        return None
    return out[-1].strip() or None


def to_linux_display_path(path: str | Path) -> str:
    raw = str(path)
    if sys.platform == "win32":
        converted = _wsl_exe_wslpath_u(raw)
        return converted or raw
    linux = windows_path_to_linux(path)
    return linux or raw


def windows_path_to_linux(path: str | Path) -> str | None:
    raw = str(path).strip()
    if not raw:
        return None
    if sys.platform == "win32":
        return raw
    if Path(raw).exists() and (raw.startswith("/") or raw.startswith("\\")):
        return str(Path(raw))
    if not is_wsl():
        return None
    wslpath = shutil.which("wslpath")
    if not wslpath:
        return None
    try:
        proc = subprocess.run(
            [wslpath, "-u", raw],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    out = proc.stdout.strip()
    return out or None


def sophon_windows_root() -> str:
    env = os.environ.get("SOPHON_WINDOWS_ROOT", "").strip()
    if env:
        return env.rstrip("\\/")
    return DEFAULT_SOPHON_WINDOWS_ROOT


def sophon_windows_root_linux() -> Path | None:
    converted = windows_path_to_linux(sophon_windows_root())
    return Path(converted) if converted else None


def windows_userprofile_linux() -> Path | None:
    if sys.platform == "win32":
        raw = os.environ.get("USERPROFILE", "").strip()
        return Path(raw) if raw else None
    if not is_wsl():
        return None
    cmd = shutil.which("cmd.exe")
    if not cmd:
        return None
    try:
        proc = subprocess.run(
            [cmd, "/c", "echo %USERPROFILE%"],
            check=True,
            capture_output=True,
            text=True,
            cwd="/mnt/c/Windows",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    win = proc.stdout.strip().splitlines()
    if not win:
        return None
    linux = windows_path_to_linux(win[-1].strip())
    return Path(linux) if linux else None


def windows_local_appdata_linux() -> Path | None:
    if sys.platform == "win32":
        raw = os.environ.get("LOCALAPPDATA", "").strip()
        return Path(raw) if raw else None
    home = windows_userprofile_linux()
    if home is None:
        return None
    local = home / "AppData" / "Local"
    return local if local.is_dir() else None


def store_pwsh_windows() -> str:
    return r"%LOCALAPPDATA%\Microsoft\WindowsApps\pwsh.exe"


def wsl_windows_venv_message() -> str:
    win_root = sophon_windows_root()
    return "\n".join(
        [
            "sophon desktop TUI requires the Windows virtual environment on the deploy tree.",
            "",
            "From WSL (copies this checkout to Windows, uv sync, Windows Terminal profile):",
            "  sophon-cli deploy-windows",
            "",
            "Or from Store PowerShell on Windows:",
            f"  cd {win_root}",
            "  $env:UV_LINK_MODE = \"copy\"",
            "  uv sync --extra tui --extra finetune",
            "",
            "Then from WSL:",
            "  sophon-cli chat --windows",
            "",
            "Stay on WSL/Linux Python instead:",
            "  sophon-cli chat --linux",
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
    env = os.environ.get("SOPHON_WINDOWS_CLI", "").strip().lower()
    if env in ("1", "true", "yes", "windows", "win"):
        return "windows"
    if env in ("0", "false", "no", "linux", "wsl"):
        return "linux"
    return "auto"


def windows_console_script_available(stem: str = "sophon-cli") -> bool:
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
        win_script = resolve_windows_console_script("sophon-chat-tui")
        if win_script is not None:
            return [str(win_script), *args]
    script = resolve_installed_script("sophon-chat-tui")
    if script is not None:
        return [str(script), *args]
    return module_fallback_argv("sophon.cli.backends.textual", args)


def desktop_terminal_available() -> bool:
    if sys.platform == "win32":
        return True
    if sys.platform == "darwin":
        return True
    if is_wsl() and resolve_windows_console_script("sophon-cli") is not None:
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
    for stem in SOPHON_CONSOLE_STEMS:
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
        print("sophon: Scripts shims already present (or bin/ entrypoints missing).", flush=True)
        return
    print("sophon: created Scripts shims for cross-platform paths:", flush=True)
    for path in created:
        print(f"  {path}", flush=True)
