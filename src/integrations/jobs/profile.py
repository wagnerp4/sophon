from __future__ import annotations

import re

_DROP_LINE = re.compile(
    r"^\s*(mobile|email|current living place|permanent parent place)\s*:",
    re.I,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_PHONE = re.compile(r"\+\d[\d\s()-]{6,}")


def redact_profile(text: str) -> str:
    kept: list[str] = []
    for raw in text.splitlines():
        if raw.strip().startswith("## Build") or "Puer Aetern" in raw:
            break
        if _DROP_LINE.match(raw):
            continue
        line = _EMAIL.sub("", raw)
        line = _PHONE.sub("", line)
        if line.strip():
            kept.append(line.rstrip())
    return "\n".join(kept).strip()
