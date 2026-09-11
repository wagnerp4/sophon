from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_TIMEOUT_S = 60.0
_MAX_OUTPUT_CHARS = 12_000


def shell_tools_enabled() -> bool:
    raw = os.environ.get("ORODRUIN_SHELL_TOOLS", "0").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def shell_timeout_s() -> float:
    raw = os.environ.get("ORODRUIN_SHELL_TIMEOUT_S", "").strip()
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
        cd = self.handle_cd_command(command)
        if cd is not None:
            return cd
        timeout = shell_timeout_s() if timeout_s is None else max(1.0, float(timeout_s))
        if sys.platform == "win32":
            return self._run_windows(command, timeout_s=timeout)
        return self._run_unix(command, timeout_s=timeout)

    def _run_windows(self, command: str, *, timeout_s: float) -> ShellResult:
        shell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
        if shell:
            argv = [shell, "-NoLogo", "-NoProfile", "-Command", command]
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
        return ShellResult(
            exit_code=int(proc.returncode),
            output=truncate_output(output.rstrip() if output.strip() else f"(exit {proc.returncode})"),
            cwd=self.cwd,
        )

    def _run_unix(self, command: str, *, timeout_s: float) -> ShellResult:
        shell = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
        try:
            code, output = self._run_pty([shell, "-lc", command], timeout_s=timeout_s)
        except Exception as exc:
            return ShellResult(exit_code=1, output=truncate_output(f"error: {exc}"), cwd=self.cwd)
        text = output.rstrip() if output.strip() else f"(exit {code})"
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
