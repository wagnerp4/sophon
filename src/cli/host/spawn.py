from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cli.host.profiles import (
    install_windows_terminal_profile,
    profile_name,
    windows_profile_installed,
)
from cli.host.session_log import make_session_log_path


def is_tui_child_process() -> bool:
    return os.environ.get("SOPHON_TUI_CHILD", "").strip() in ("1", "true", "yes")


def _strip_spawn_flags(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    for arg in argv:
        if arg in ("--spawn-window", "--no-spawn-window"):
            continue
        cleaned.append(arg)
    return cleaned


def project_root() -> Path:
    try:
        from utils.device.env_bootstrap import sophon_project_root

        return sophon_project_root()
    except Exception:
        return Path.cwd().resolve()


def child_runtime_env(base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    root = project_root()
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
    from utils.device.platform import chat_tui_argv

    args = _strip_spawn_flags(list(user_argv))
    if args and args[0] == "chat":
        args = args[1:]
    args = [arg for arg in args if arg != "--tui"]
    if "--no-spawn-window" not in args:
        args = ["--no-spawn-window", *args]
    return chat_tui_argv(args)


def _windows_cmd_line(
    command: list[str],
    env: dict[str, str],
    *,
    utf8_console: bool,
) -> str:
    parts: list[str] = []
    if utf8_console:
        parts.append("chcp 65001 >nul")
    from utils.device.platform import child_env_keys

    for key in child_env_keys():
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
    use_sophon_profile: bool,
    utf8_console: bool,
) -> subprocess.Popen[bytes]:
    wt = shutil.which("wt") or shutil.which("wt.exe")
    if wt:
        profile = "sophon chat"
        use_profile = use_sophon_profile
        if use_profile:
            try:
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
    env_prefix = " ".join(f'{key}="{env[key]}"' for key in ("SOPHON_TUI_LOG", "SOPHON_TUI_CHILD") if key in env)
    cmd = " ".join(shlex.quote(part) for part in command)
    shell_cmd = f"cd {shlex.quote(cwd)}; {env_prefix} {cmd}"
    script = f'tell application "Terminal" to do script {shlex.quote(shell_cmd)}'
    return subprocess.Popen(["osascript", "-e", script], env=env, cwd=cwd)


def _spawn_linux(
    command: list[str],
    env: dict[str, str],
    cwd: str,
    *,
    window_title: str = "sophon chat",
) -> subprocess.Popen[bytes]:
    from utils.device.platform import is_wsl, sophon_windows_root

    if is_wsl():
        cmd = shutil.which("cmd.exe")
        if cmd:
            return subprocess.Popen(
                [
                    cmd,
                    "/c",
                    "start",
                    "wt.exe",
                    "-w",
                    "0",
                    "nt",
                    "-p",
                    profile_name(),
                    "-d",
                    sophon_windows_root(),
                ],
                env=env,
                cwd="/mnt/c/Windows",
            )
    raise RuntimeError(
        "Linux desktop TUI spawn needs Windows Terminal from WSL. "
        "Run sophon-cli deploy-windows, then sophon-cli chat --windows, "
        "or use --no-spawn-window."
    )


def spawn_desktop_terminal(
    command: list[str],
    env: dict[str, str],
    cwd: str,
    *,
    window_title: str = "sophon chat",
    use_sophon_profile: bool = True,
    utf8_console: bool = True,
) -> subprocess.Popen[bytes]:
    if sys.platform == "win32":
        return _spawn_windows(
            command,
            env,
            cwd,
            window_title=window_title,
            use_sophon_profile=use_sophon_profile,
            utf8_console=utf8_console,
        )
    if sys.platform == "darwin":
        return _spawn_macos(command, env, cwd)
    return _spawn_linux(command, env, cwd, window_title=window_title)


def _print_parent_header(log_path: Path, pid: int) -> None:
    print(f"[sophon] Desktop TUI launched (pid={pid})", flush=True)
    print(f"[sophon] Log file: {log_path}", flush=True)
    print("[sophon] --- mirrored session log ---", flush=True)


def _drain_log(log_path: Path, offset: int) -> int:
    if not log_path.exists():
        return offset
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            chunk = handle.read()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            return handle.tell()
    except OSError as exc:
        print(f"[sophon] Log read failed ({log_path}): {exc}", flush=True)
        return offset


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


def _read_child_pid(
    log_path: Path,
    *,
    log_offset_at_launch: int = 0,
    timeout_s: float = 120.0,
) -> int | None:
    pattern = re.compile(r"child pid=(\d+)")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(log_offset_at_launch)
                    chunk = handle.read()
            except OSError:
                time.sleep(0.1)
                continue
            matches = list(pattern.finditer(chunk))
            if matches:
                return int(matches[-1].group(1))
        time.sleep(0.1)
    return None


def tail_log_until_exit(
    log_path: Path,
    _launcher: subprocess.Popen[bytes],
    *,
    log_offset_at_launch: int = 0,
) -> int:
    offset = log_offset_at_launch
    child_pid = _read_child_pid(log_path, log_offset_at_launch=log_offset_at_launch)
    if child_pid is None:
        print("[sophon] Timed out waiting for desktop TUI process.", flush=True)
        return 1

    print(f"[sophon] Tracking desktop TUI pid={child_pid}", flush=True)
    while _process_running(child_pid):
        offset = _drain_log(log_path, offset)
        time.sleep(0.15)

    offset = _drain_log(log_path, offset)
    print("[sophon] Desktop TUI closed.", flush=True)
    return 0


def launch_tui_in_new_terminal(user_argv: list[str]) -> int:
    from utils.device.env_bootstrap import load_sophon_dotenv

    load_sophon_dotenv()
    log_path = make_session_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_offset_at_launch = log_path.stat().st_size if log_path.is_file() else 0

    env = child_runtime_env(os.environ.copy())
    env["SOPHON_TUI_LOG"] = str(log_path.resolve())
    env["SOPHON_TUI_CHILD"] = "1"

    command = build_child_argv(user_argv)
    cwd = str(project_root())
    try:
        if sys.platform == "win32" and not windows_profile_installed():
            installed = install_windows_terminal_profile()
            if installed is not None:
                print(f"[sophon] Installed Windows Terminal profile: {installed}", flush=True)
    except Exception:
        pass

    try:
        proc = spawn_desktop_terminal(command, env, cwd)
    except RuntimeError as exc:
        print(f"[sophon] {exc}", flush=True)
        return 1
    _print_parent_header(log_path.resolve(), proc.pid)
    return tail_log_until_exit(
        log_path.resolve(),
        proc,
        log_offset_at_launch=log_offset_at_launch,
    )
