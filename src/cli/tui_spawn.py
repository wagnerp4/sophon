from __future__ import annotations

import datetime as _dt
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO


def _project_log_dir() -> Path:
    """TODO:replace and move to src/utils/logging.py"""
    from utils.device.env_bootstrap import mithril_chat_logs_dir

    return mithril_chat_logs_dir()


def make_session_log_path() -> Path:
    """TODO:replace and move to src/utils/logging.py"""
    return _project_log_dir() / "tui.log"


class TuiSessionLog:
    """TODO:replace and move to src/utils/logging.py"""
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = open(self.path, "a", encoding="utf-8")
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write("meta", f"--- session start {stamp} ---")

    def write(self, prefix: str, text: str) -> None:
        if self._file.closed:
            return
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for line in text.splitlines() or [""]:
            self._file.write(f"[{stamp}] [{prefix}] {line}\n")
        self._file.flush()

    def close(self) -> None:
        try:
            if not self._file.closed:
                stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.write("meta", f"--- session end {stamp} ---")
                self._file.close()
        except Exception:
            pass


class _TeeTextIO:
    """TODO:replace and move to src/utils/logging.py"""
    def __init__(self, stream: TextIO, log: TuiSessionLog, prefix: str) -> None:
        self._stream = stream
        self._log = log
        self._prefix = prefix

    def write(self, data: str) -> int:
        if data:
            stripped = data.rstrip("\n\r")
            if stripped:
                self._log.write(self._prefix, stripped)
        return self._stream.write(data)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def open_session_log_from_env() -> TuiSessionLog | None:
    raw = os.environ.get("MITHRIL_TUI_LOG", "").strip()
    if not raw:
        return None
    return TuiSessionLog(Path(raw))


def is_tui_child_process() -> bool:
    return os.environ.get("MITHRIL_TUI_CHILD", "").strip() in ("1", "true", "yes")


def _strip_spawn_flags(argv: list[str]) -> list[str]:
    skip_next = False
    cleaned: list[str] = []
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--spawn-window", "--no-spawn-window"):
            continue
        cleaned.append(arg)
    return cleaned


def _project_root() -> Path:
    try:
        from utils.device.env_bootstrap import mithril_project_root

        return mithril_project_root()
    except Exception:
        return Path.cwd().resolve()


def _child_runtime_env(base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    root = _project_root()
    src = str((root / "src").resolve())
    parts = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    if src not in parts:
        parts.insert(0, src)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if sys.platform == "win32":
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def build_child_argv(user_argv: list[str]) -> list[str]:
    args = _strip_spawn_flags(list(user_argv))
    if args and args[0] == "chat":
        args = args[1:]
    args = [arg for arg in args if arg != "--tui"]
    if "--no-spawn-window" not in args:
        args = ["--no-spawn-window", *args]

    scripts_dir = Path(sys.executable).resolve().parent
    for script_name in ("mithril-chat-tui.exe", "mithril-chat-tui"):
        script_path = scripts_dir / script_name
        if script_path.is_file():
            return [str(script_path), *args]

    return [sys.executable, "-m", "mithril.cli.textual_app", *args]


def _windows_cmd_line(
    command: list[str],
    env: dict[str, str],
    *,
    utf8_console: bool,
) -> str:
    parts: list[str] = []
    if utf8_console:
        parts.append("chcp 65001 >nul")
    for key in (
        "MITHRIL_TUI_LOG",
        "MITHRIL_TUI_CHILD",
        "MITHRIL_INTERFACE",
        "PYTHONPATH",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
    ):
        value = env.get(key)
        if value:
            parts.append(f"set \"{key}={value}\"")
    parts.append(subprocess.list2cmdline(command))
    return " && ".join(parts)


def _spawn_windows(
    command: list[str],
    env: dict[str, str],
    cwd: str,
    *,
    window_title: str,
    use_mithril_profile: bool,
    utf8_console: bool,
) -> subprocess.Popen[bytes]:
    wt = shutil.which("wt") or shutil.which("wt.exe")
    if wt:
        profile = "mithril chat"
        use_profile = use_mithril_profile
        if use_profile:
            try:
                from cli.terminal_profiles import profile_name, windows_profile_installed

                use_profile = windows_profile_installed()
                profile = profile_name()
            except Exception:
                use_profile = False

        cmd_line = _windows_cmd_line(command, env, utf8_console=utf8_console)
        if use_profile:
            wt_args = [wt, "-w", "0", "nt", "-p", profile, "-d", cwd, "cmd", "/c", cmd_line]
        else:
            wt_args = [
                wt,
                "-w",
                "0",
                "nt",
                "--title",
                window_title,
                "-d",
                cwd,
                "cmd",
                "/c",
                cmd_line,
            ]
        return subprocess.Popen(wt_args, env=env, cwd=cwd)

    create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return subprocess.Popen(
        command,
        env=env,
        cwd=cwd,
        creationflags=create_new_console,
    )


def _spawn_macos(command: list[str], env: dict[str, str], cwd: str) -> subprocess.Popen[bytes]:
    env_prefix = " ".join(f'{key}="{env[key]}"' for key in ("MITHRIL_TUI_LOG", "MITHRIL_TUI_CHILD") if key in env)
    cmd = " ".join(shlex.quote(part) for part in command)
    shell_cmd = f"cd {shlex.quote(cwd)}; {env_prefix} {cmd}"
    script = f'tell application "Terminal" to do script {shlex.quote(shell_cmd)}'
    return subprocess.Popen(["osascript", "-e", script], env=env, cwd=cwd)


def _spawn_linux(command: list[str], env: dict[str, str], cwd: str) -> subprocess.Popen[bytes]:
    candidates = [
        ["gnome-terminal", "--wait", "--"],
        ["konsole", "-e"],
        ["xfce4-terminal", "-e"],
        ["xterm", "-e"],
    ]
    for prefix in candidates:
        if shutil.which(prefix[0]):
            args = [*prefix, *command]
            return subprocess.Popen(args, env=env, cwd=cwd)
    create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    if create_new_console:
        return subprocess.Popen(command, env=env, cwd=cwd, creationflags=create_new_console)
    return subprocess.Popen(command, env=env, cwd=cwd)


def spawn_desktop_terminal(
    command: list[str],
    env: dict[str, str],
    cwd: str,
    *,
    window_title: str = "mithril chat",
    use_mithril_profile: bool = True,
    utf8_console: bool = True,
) -> subprocess.Popen[bytes]:
    if sys.platform == "win32":
        return _spawn_windows(
            command,
            env,
            cwd,
            window_title=window_title,
            use_mithril_profile=use_mithril_profile,
            utf8_console=utf8_console,
        )
    if sys.platform == "darwin":
        return _spawn_macos(command, env, cwd)
    return _spawn_linux(command, env, cwd)


def _print_parent_header(log_path: Path, pid: int) -> None:
    print(f"[mithril] Desktop TUI launched (pid={pid})", flush=True)
    print(f"[mithril] Log file: {log_path}", flush=True)
    print("[mithril] --- mirrored session log ---", flush=True)


def _drain_log(log_path: Path, offset: int) -> int:
    if not log_path.exists():
        return offset
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        chunk = handle.read()
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        return handle.tell()


def _process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        process_query_limited = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_child_pid(log_path: Path, timeout_s: float = 120.0) -> int | None:
    pattern = re.compile(r"child pid=(\d+)")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            match = pattern.search(text)
            if match:
                return int(match.group(1))
        time.sleep(0.1)
    return None


def tail_log_until_exit(log_path: Path, _launcher: subprocess.Popen[bytes]) -> int:
    offset = 0
    child_pid = _read_child_pid(log_path)
    if child_pid is None:
        print("[mithril] Timed out waiting for desktop TUI process.", flush=True)
        return 1

    print(f"[mithril] Tracking desktop TUI pid={child_pid}", flush=True)
    while _process_running(child_pid):
        offset = _drain_log(log_path, offset)
        time.sleep(0.15)

    offset = _drain_log(log_path, offset)
    print("[mithril] Desktop TUI closed.", flush=True)
    return 0


def launch_tui_in_new_terminal(user_argv: list[str]) -> int:
    log_path = make_session_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = _child_runtime_env(os.environ.copy())
    env["MITHRIL_TUI_LOG"] = str(log_path.resolve())
    env["MITHRIL_TUI_CHILD"] = "1"

    command = build_child_argv(user_argv)
    cwd = str(_project_root())
    try:
        from cli.terminal_profiles import install_windows_terminal_profile, windows_profile_installed

        if sys.platform == "win32" and not windows_profile_installed():
            installed = install_windows_terminal_profile()
            if installed is not None:
                print(f"[mithril] Installed Windows Terminal profile: {installed}", flush=True)
    except Exception:
        pass

    proc = spawn_desktop_terminal(command, env, cwd)
    _print_parent_header(log_path.resolve(), proc.pid)
    return tail_log_until_exit(log_path.resolve(), proc)
