from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Captured:
    code: int
    text: str
    stdout: str = ""
    stderr: str = ""


class WslHost:
    def __init__(self, linux_root: str, distro: str) -> None:
        self.linux_root = linux_root.rstrip("/") or "."
        self.distro = distro or "Debian"

    def on_windows(self) -> bool:
        return sys.platform == "win32"

    def argv(self, command: str) -> list[str]:
        if self.on_windows():
            wsl = shutil.which("wsl.exe") or "wsl.exe"
            return [
                wsl,
                "-d",
                self.distro,
                "--cd",
                self.linux_root,
                "--",
                "bash",
                "-lc",
                command,
            ]
        return ["bash", "-lc", command]

    def run_capture(self, command: str, timeout: float = 45.0) -> Captured:
        kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if not self.on_windows():
            kwargs["cwd"] = self.linux_root
        try:
            proc = subprocess.run(self.argv(command), **kwargs)
        except FileNotFoundError as exc:
            return Captured(code=127, text=str(exc), stdout="", stderr=str(exc))
        except subprocess.TimeoutExpired as exc:
            out = str(exc.stdout or "")
            err = str(exc.stderr or "")
            return Captured(code=124, text=out or err or "timeout", stdout=out, stderr=err)
        out = proc.stdout or ""
        err = proc.stderr or ""
        return Captured(code=int(proc.returncode), text=out + err, stdout=out, stderr=err)

    def popen(self, command: str) -> subprocess.Popen:
        kwargs: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if not self.on_windows():
            kwargs["cwd"] = self.linux_root
        if self.on_windows():
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return subprocess.Popen(self.argv(command), **kwargs)

    def disk_bytes(self) -> tuple[int | None, int | None]:
        captured = self.run_capture("df -P -B1 . | awk 'NR==2 {print $2, $4}'", timeout=15.0)
        parts = captured.text.strip().split()
        if captured.code != 0 or len(parts) < 2:
            return None, None
        try:
            total = int(parts[0])
            avail = int(parts[1])
        except ValueError:
            return None, None
        return total, avail

    def path_exists(self, rel: str) -> bool:
        token = rel.replace("\"", "\\\"")
        captured = self.run_capture(f"test -e \"{token}\" && echo present || echo missing", timeout=10.0)
        return "present" in captured.text


def stream_popen(proc: subprocess.Popen, on_log, *, limit: int = 0) -> int:
    n = 0
    if proc.stdout is None:
        return int(proc.wait())
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        on_log(line)
        n += 1
        if limit > 0 and n >= limit:
            break
    return int(proc.wait())
