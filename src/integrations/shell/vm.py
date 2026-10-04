from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from integrations.shell.sandbox import popen_captured, to_wsl_path

VM_HOME = "/home/sophon/work"


def _script_path() -> Path:
    from utils.device.env_bootstrap import sophon_project_root

    return Path(sophon_project_root()) / "scripts" / "sandbox-vm.sh"


def _argv(*args: str) -> list[str]:
    script = _script_path()
    if sys.platform == "win32":
        distro = os.environ.get("SOPHON_WSL_DISTRO", "Debian")
        return ["wsl.exe", "-d", distro, "--", "bash", to_wsl_path(script), *args]
    return ["bash", str(script), *args]


def vm_control(action: str, *extra: str, timeout_s: float = 120.0) -> tuple[int, str]:
    return popen_captured(_argv(action, *extra), cwd=_script_path().parent, timeout_s=timeout_s)


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def vm_exec(command: str, *, cwd: str = VM_HOME, timeout_s: float = 60.0) -> tuple[int, str]:
    inner = max(1, int(timeout_s))
    return popen_captured(
        _argv("exec-b64", str(inner), _b64(cwd), _b64(command)),
        cwd=_script_path().parent,
        timeout_s=inner + 30,
    )
