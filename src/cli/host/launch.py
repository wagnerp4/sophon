"""The directory the user launched sophon from, carried across every spawn route (sophon-010).

The path travels as a base64url token (``[A-Za-z0-9_-]``), never as raw text, so no
command line (cmd.exe, Windows Terminal, POSIX shell, AppleScript) can misparse it.
Same-version routes pass it as the hidden ``--launch-dir-token`` option; the WSL to
Windows re-exec passes it in ``SOPHON_LAUNCH_DIR_TOKEN`` via ``WSLENV`` so an older
Windows deploy simply ignores it.

Always import this module as bare ``cli.host.launch``: the recorded value is a
process global and must not be split across two module objects.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from utils.device.launch_token import (
    TOKEN_ENV,
    TOKEN_OPTION,
    decode_token,
    encode_token,
)

__all__ = [
    "TOKEN_ENV",
    "TOKEN_OPTION",
    "decode_token",
    "encode_token",
    "escape_controls",
    "launch_dir",
    "markup_safe_path",
    "record_launch_dir",
    "token_from_argv",
    "with_token_arg",
]

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_recorded: Path | None = None
_warned: set[str] = set()


def _warn(message: str) -> None:
    if message in _warned:
        return
    _warned.add(message)
    print(f"[sophon] {message}", file=sys.stderr, flush=True)


def _valid_dir(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_absolute() and path.is_dir()
    except (OSError, ValueError):
        return False


def _is_windows_system_dir(path: Path) -> bool:
    root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or r"C:\Windows"

    def norm(value: str) -> str:
        return os.path.normcase(value).rstrip("\\/").lower()

    return norm(str(path)) == norm(root)


def record_launch_dir(token: str | None = None) -> Path:
    """Record the launch dir once per process: argv token > env token (Windows) > cwd."""
    global _recorded
    env_token = os.environ.pop(
        TOKEN_ENV, None
    )  # always popped so children never inherit it
    if _recorded is not None:
        return _recorded
    candidates: list[tuple[str, str]] = []
    if token:
        candidates.append(("argument", token))
    if env_token and sys.platform == "win32":
        candidates.append(("environment", env_token))
    chosen: Path | None = None
    for source, value in candidates:
        decoded = decode_token(value)
        if _valid_dir(decoded):
            chosen = decoded
            break
        _warn(f"ignoring invalid launch-dir token from {source}")
    if chosen is None:
        chosen = Path.cwd()
    if sys.platform == "win32" and _is_windows_system_dir(chosen):
        from utils.device.env_bootstrap import sophon_project_root

        chosen = sophon_project_root()
    try:
        _recorded = chosen.resolve()
    except OSError:
        _recorded = chosen.absolute()
    return _recorded


def launch_dir() -> Path:
    """The recorded launch dir; records from the cwd if nothing was recorded yet."""
    return _recorded if _recorded is not None else record_launch_dir()


def token_from_argv(argv: list[str]) -> str | None:
    """The last --launch-dir-token value in argv, if any."""
    found: str | None = None
    for index, arg in enumerate(argv):
        if arg == TOKEN_OPTION and index + 1 < len(argv):
            found = argv[index + 1]
        elif arg.startswith(TOKEN_OPTION + "="):
            found = arg[len(TOKEN_OPTION) + 1 :]
    return found


def with_token_arg(argv: list[str], path: Path) -> list[str]:
    """argv without any --launch-dir-token, plus one token for path (none if unencodable)."""
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == TOKEN_OPTION:
            skip = True
            continue
        if arg.startswith(TOKEN_OPTION + "="):
            continue
        out.append(arg)
    token = encode_token(path)
    if token is None:
        _warn("launch dir cannot be encoded; the new window opens in the install root")
        return out
    return [*out, TOKEN_OPTION, token]


def escape_controls(text: str) -> str:
    """Replace control characters with \\xNN so a path cannot fake lines or styles."""
    return _CONTROL_RE.sub(lambda match: f"\\x{ord(match.group()):02x}", text)


def markup_safe_path(path: Path | str) -> str:
    """A launch-dir-derived path made safe for Rich markup widgets."""
    from rich.markup import escape

    return escape(escape_controls(str(path)))


def _reset_for_tests() -> None:
    global _recorded
    _recorded = None
    _warned.clear()
