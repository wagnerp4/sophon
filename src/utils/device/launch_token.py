"""Base64url codec for the launch-dir token (sophon-010). Pure; shared by cli and utils.

The alphabet [A-Za-z0-9_-] has no metacharacter for cmd.exe, Windows Terminal, POSIX
shells or AppleScript, so a path never appears as text on any command line.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
from pathlib import Path

TOKEN_OPTION = "--launch-dir-token"
TOKEN_ENV = "SOPHON_LAUNCH_DIR_TOKEN"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{1,8192}")


def encode_token(path: Path | str) -> str | None:
    """Encode a path as an unpadded base64url token; None if the path cannot be encoded."""
    try:
        raw = os.fsencode(path)
    except (TypeError, ValueError, UnicodeError):
        return None
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_token(token: str) -> Path | None:
    """Decode a token from encode_token; None for anything malformed."""
    if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
        return None
    padded = token.translate(str.maketrans("-_", "+/")) + "=" * (-len(token) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
        text = os.fsdecode(raw)
    except (ValueError, binascii.Error, UnicodeError):
        return None
    if not text or "\x00" in text:
        return None
    return Path(text)
