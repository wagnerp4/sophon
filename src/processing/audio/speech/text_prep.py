from __future__ import annotations

import re


def plain_text_for_tts(text: str) -> str:
    t = re.sub(r"```[\s\S]*?```", " ", text)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    return " ".join(t.split())


def truncate_for_spoken_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip(), True
