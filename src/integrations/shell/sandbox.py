from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

SANDBOX_MODES = ("read-only", "workspace-write", "full", "vm")


def normalize_sandbox(value: object) -> str:
    token = str(value or "full").strip().lower().replace("_", "-")
    if token in ("readonly", "read-only"):
        return "read-only"
    if token in ("workspace-write", "workspace"):
        return "workspace-write"
    if token in ("vm", "qemu"):
        return "vm"
    return "full"


def protected_paths(workspace: Path) -> list[Path]:
    root = workspace / ".sophon"
    names = (
        root / "harness.yaml",
        root / "harness.local.yaml",
        workspace / ".env",
    )
    return [path for path in names if path.exists()]


def to_wsl_path(path: Path) -> str:
    text = str(path.resolve())
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        rest = text[2:].replace("\\", "/")
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return text.replace("\\", "/")


def bwrap_argv(
    command: str,
    *,
    workspace: Path,
    cwd: Path,
    mode: str,
    path_map,
) -> list[str]:
    ws = path_map(workspace)
    here = path_map(cwd)
    argv = ["bwrap", "--die-with-parent", "--unshare-net", "--ro-bind", "/", "/"]
    if mode == "workspace-write":
        argv.extend(["--bind", ws, ws])
        for path in protected_paths(workspace):
            mapped = path_map(path)
            argv.extend(["--ro-bind", mapped, mapped])
    argv.extend(["--chdir", here, "bash", "-lc", command])
    return argv


def _which_bwrap_wsl() -> bool:
    cached = getattr(_which_bwrap_wsl, "cached", None)
    if cached is not None:
        return bool(cached)
    try:
        proc = subprocess.run(
            ["wsl.exe", "-d", "Debian", "--", "which", "bwrap"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        ok = proc.returncode == 0 and bool((proc.stdout or "").strip())
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    setattr(_which_bwrap_wsl, "cached", ok)
    return ok


def windows_confined_argv(command: str, *, workspace: Path, cwd: Path, mode: str) -> list[str] | str:
    if not _which_bwrap_wsl():
        return "error: sandbox unavailable"
    inner = bwrap_argv(command, workspace=workspace, cwd=cwd, mode=mode, path_map=to_wsl_path)
    script = " ".join(_sh_quote(part) for part in inner)
    return ["wsl.exe", "-d", "Debian", "--", "bash", "-lc", script]


def _sh_quote(part: str) -> str:
    return "'" + part.replace("'", "'\"'\"'") + "'"


def linux_confined_argv(command: str, *, workspace: Path, cwd: Path, mode: str) -> list[str] | str:
    if landlock_supported():
        return ["__landlock__", command]
    bwrap = shutil.which("bwrap")
    if not bwrap:
        return "error: sandbox unavailable"
    argv = bwrap_argv(
        command,
        workspace=workspace,
        cwd=cwd,
        mode=mode,
        path_map=lambda path: str(path.resolve()),
    )
    argv[0] = bwrap
    return argv


def landlock_supported() -> bool:
    if sys.platform == "win32":
        return False
    cached = getattr(landlock_supported, "cached", None)
    if cached is not None:
        return bool(cached)
    ok = _probe_landlock()
    setattr(landlock_supported, "cached", ok)
    return ok


def _probe_landlock() -> bool:
    try:
        fd = _landlock_create(handled=_READ_BITS)
    except OSError:
        return False
    if fd < 0:
        return False
    os.close(fd)
    return True


_READ_BITS = (1 << 0) | (1 << 2) | (1 << 3)
_WRITE_BITS = _READ_BITS | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 12)


def _syscall(number: int, *args: int) -> int:
    libc = __import__("ctypes").CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = __import__("ctypes").c_long
    result = int(syscall(number, *args))
    if result < 0:
        err = __import__("ctypes").get_errno()
        raise OSError(err, os.strerror(err))
    return result


def _landlock_numbers() -> tuple[int, int, int]:
    machine = os.uname().machine
    if machine in ("x86_64", "amd64"):
        return 444, 445, 446
    if machine in ("aarch64", "arm64"):
        return 444, 445, 446
    return 444, 445, 446


def _landlock_create(handled: int) -> int:
    import ctypes

    class Attr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    create, _add, _restrict = _landlock_numbers()
    attr = Attr(handled)
    return _syscall(create, ctypes.addressof(attr), ctypes.sizeof(attr), 0)


def apply_landlock(workspace: Path, mode: str) -> None:
    import ctypes

    class PathAttr(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    _create, add_rule, restrict_self = _landlock_numbers()
    handled = _WRITE_BITS if mode == "workspace-write" else _READ_BITS
    ruleset = _landlock_create(handled=handled)
    try:
        root_fd = os.open("/", os.O_PATH | os.O_CLOEXEC)
        try:
            attr = PathAttr(_READ_BITS, root_fd)
            _syscall(add_rule, ruleset, 1, ctypes.addressof(attr), 0)
        finally:
            os.close(root_fd)
        if mode == "workspace-write":
            ws_fd = os.open(str(workspace), os.O_PATH | os.O_CLOEXEC)
            try:
                attr = PathAttr(_WRITE_BITS, ws_fd)
                _syscall(add_rule, ruleset, 1, ctypes.addressof(attr), 0)
            finally:
                os.close(ws_fd)
        _syscall(restrict_self, ruleset, 0)
    finally:
        os.close(ruleset)


def popen_captured(argv: list[str], *, cwd: Path, timeout_s: float, job: bool = False) -> tuple[int, str]:
    if job and sys.platform == "win32":
        return _run_windows_job(argv, cwd=cwd, timeout_s=timeout_s)
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return 124, f"error: command timed out after {timeout_s:.0f}s"
    except Exception as exc:
        return 1, f"error: {exc}"
    output = (proc.stdout or "") + (proc.stderr or "")
    text = output.rstrip() if output.strip() else f"(exit {proc.returncode})"
    return int(proc.returncode), text


def _run_windows_job(argv: list[str], *, cwd: Path, timeout_s: float) -> tuple[int, str]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if job:
            kernel32.AssignProcessToJobObject(job, ctypes.c_void_p(int(proc._handle)))
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=2)
            except Exception:
                pass
            return 124, f"error: command timed out after {timeout_s:.0f}s"
        output = (stdout or "") + (stderr or "")
        text = output.rstrip() if output.strip() else f"(exit {proc.returncode})"
        return int(proc.returncode or 0), text
    finally:
        if job:
            kernel32.CloseHandle(job)


def confined_preexec(workspace: Path, mode: str):
    def _inner() -> None:
        apply_landlock(workspace, mode)

    return _inner
