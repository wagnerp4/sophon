from __future__ import annotations

import os
import shutil
import subprocess
import sys
import base64
from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_TIMEOUT_S = 60.0
_MAX_OUTPUT_CHARS = 12_000


def shell_tools_enabled() -> bool:
    raw = os.environ.get("SOPHON_SHELL_TOOLS", "0").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def shell_timeout_s() -> float:
    raw = os.environ.get("SOPHON_SHELL_TIMEOUT_S", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT_S


def truncate_output(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


@dataclass
class ShellResult:
    exit_code: int
    output: str
    cwd: Path
    handled_as_cd: bool = False


@dataclass
class ShellSession:
    cwd: Path = field(default_factory=Path.cwd)

    def resolve_path(self, path: str | None) -> Path:
        raw = (path or "").strip().strip('"').strip("'")
        if not raw or raw in (".",):
            return self.cwd.resolve()
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = self.cwd / target
        return target.resolve()

    def change_directory(self, path: str | None) -> tuple[bool, str]:
        if path is None or str(path).strip() == "":
            self.cwd = Path.home().resolve()
            return True, str(self.cwd)
        target = self.resolve_path(path)
        if not target.is_dir():
            return False, f"not a directory: {target}"
        self.cwd = target
        return True, str(self.cwd)

    def handle_cd_command(self, command: str) -> ShellResult | None:
        parts = command.strip().split(maxsplit=1)
        if not parts or parts[0].lower() not in ("cd", "set-location"):
            return None
        arg = parts[1] if len(parts) > 1 else None
        ok, msg = self.change_directory(arg)
        if ok:
            return ShellResult(exit_code=0, output=f"cwd -> {msg}", cwd=self.cwd, handled_as_cd=True)
        return ShellResult(exit_code=1, output=msg, cwd=self.cwd, handled_as_cd=True)

    def run(self, command: str, *, timeout_s: float | None = None) -> ShellResult:
        timeout = shell_timeout_s() if timeout_s is None else max(1.0, float(timeout_s))
        mode = str(getattr(self, "sandbox_mode", "full") or "full")
        if mode == "vm":
            return self._run_vm(command, timeout_s=timeout)
        cd = self.handle_cd_command(command)
        if cd is not None:
            return cd
        if mode in ("read-only", "workspace-write"):
            return self._run_sandboxed(command, timeout_s=timeout, mode=mode)
        if sys.platform == "win32":
            return self._run_windows(command, timeout_s=timeout)
        return self._run_unix(command, timeout_s=timeout)

    def _run_vm(self, command: str, *, timeout_s: float) -> ShellResult:
        from integrations.shell.vm import VM_HOME, vm_exec

        vm_cwd = str(getattr(self, "vm_cwd", "") or VM_HOME)
        parts = command.strip().split(maxsplit=1)
        if parts and parts[0] == "cd":
            target = parts[1] if len(parts) > 1 else "~"
            code, output = vm_exec(f"cd {target} && pwd", cwd=vm_cwd, timeout_s=15)
            if code == 0 and output.strip():
                self.vm_cwd = output.strip().splitlines()[-1]
                return ShellResult(exit_code=0, output=f"vm cwd -> {self.vm_cwd}", cwd=self.cwd, handled_as_cd=True)
            return ShellResult(exit_code=code or 1, output=truncate_output(output), cwd=self.cwd, handled_as_cd=True)
        code, output = vm_exec(command, cwd=vm_cwd, timeout_s=timeout_s)
        return ShellResult(exit_code=code, output=truncate_output(output), cwd=self.cwd)

    def _run_sandboxed(self, command: str, *, timeout_s: float, mode: str) -> ShellResult:
        from integrations.shell.sandbox import (
            confined_preexec,
            landlock_supported,
            linux_confined_argv,
            popen_captured,
            windows_confined_argv,
        )

        workspace = Path(getattr(self, "sandbox_workspace", self.cwd) or self.cwd)
        if sys.platform == "win32":
            argv = windows_confined_argv(command, workspace=workspace, cwd=self.cwd, mode=mode)
            if isinstance(argv, str):
                return ShellResult(exit_code=1, output=argv, cwd=self.cwd)
            code, output = popen_captured(argv, cwd=self.cwd, timeout_s=timeout_s, job=True)
            return ShellResult(exit_code=code, output=truncate_output(output), cwd=self.cwd)
        if landlock_supported():
            shell = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
            try:
                proc = subprocess.run(
                    [shell, "-lc", command],
                    cwd=str(self.cwd),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    check=False,
                    timeout=timeout_s,
                    preexec_fn=confined_preexec(workspace, mode),
                )
            except subprocess.TimeoutExpired:
                return ShellResult(
                    exit_code=124,
                    output=truncate_output(f"error: command timed out after {timeout_s:.0f}s"),
                    cwd=self.cwd,
                )
            except Exception as exc:
                return ShellResult(exit_code=1, output=truncate_output(f"error: {exc}"), cwd=self.cwd)
            output = (proc.stdout or "") + (proc.stderr or "")
            text = output.rstrip() if output.strip() else f"(exit {proc.returncode})"
            return ShellResult(exit_code=int(proc.returncode), output=truncate_output(text), cwd=self.cwd)
        argv = linux_confined_argv(command, workspace=workspace, cwd=self.cwd, mode=mode)
        if isinstance(argv, str):
            return ShellResult(exit_code=1, output=argv, cwd=self.cwd)
        if argv and argv[0] == "__landlock__":
            return ShellResult(exit_code=1, output="error: sandbox unavailable", cwd=self.cwd)
        code, output = popen_captured(argv, cwd=self.cwd, timeout_s=timeout_s, job=False)
        return ShellResult(exit_code=code, output=truncate_output(output), cwd=self.cwd)

    def _run_windows(self, command: str, *, timeout_s: float) -> ShellResult:
        shell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
        if shell:
            encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
            argv = [shell, "-NoLogo", "-NoProfile", "-EncodedCommand", encoded]
        else:
            argv = ["cmd.exe", "/c", command]
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.cwd),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ShellResult(
                exit_code=124,
                output=truncate_output(f"error: command timed out after {timeout_s:.0f}s"),
                cwd=self.cwd,
            )
        except Exception as exc:
            return ShellResult(exit_code=1, output=truncate_output(f"error: {exc}"), cwd=self.cwd)
        output = (proc.stdout or "") + (proc.stderr or "")
        text = output.rstrip() if output.strip() else f"(exit {proc.returncode})"
        if not text.startswith("exit "):
            text = f"exit {proc.returncode}\n{text}"
        return ShellResult(
            exit_code=int(proc.returncode),
            output=truncate_output(text),
            cwd=self.cwd,
        )

    def _run_unix(self, command: str, *, timeout_s: float) -> ShellResult:
        shell = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
        try:
            code, output = self._run_pty([shell, "-lc", command], timeout_s=timeout_s)
        except Exception as exc:
            return ShellResult(exit_code=1, output=truncate_output(f"error: {exc}"), cwd=self.cwd)
        text = output.rstrip() if output.strip() else f"(exit {code})"
        if not text.startswith("exit "):
            text = f"exit {code}\n{text}"
        return ShellResult(exit_code=int(code), output=truncate_output(text), cwd=self.cwd)

    def _run_pty(self, argv: list[str], *, timeout_s: float) -> tuple[int, str]:
        import pty
        import select
        import time

        master_fd, slave_fd = pty.openpty()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.cwd),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
            )
            os.close(slave_fd)
            chunks: list[bytes] = []
            deadline = time.monotonic() + timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        pass
                    raise TimeoutError(f"command timed out after {timeout_s:.0f}s")
                ready, _, _ = select.select([master_fd], [], [], min(0.1, remaining))
                if master_fd in ready:
                    try:
                        data = os.read(master_fd, 4096)
                    except OSError:
                        data = b""
                    if data:
                        chunks.append(data)
                    else:
                        break
                if proc.poll() is not None and not ready:
                    break
            return proc.wait(), b"".join(chunks).decode("utf-8", errors="replace")
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass
