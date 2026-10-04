from __future__ import annotations

import json


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    block = text[4:end]
    body = text[end + 4 :]
    if body.startswith("\n"):
        body = body[1:]
    data: dict[str, object] = {}
    lines = block.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        if not raw.strip() or raw.startswith("  - "):
            index += 1
            continue
        if ":" not in raw:
            index += 1
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            items: list[str] = []
            nxt = index + 1
            while nxt < len(lines) and lines[nxt].startswith("  - "):
                items.append(_unquote(lines[nxt][4:].strip()))
                nxt += 1
            if items:
                data[key] = items
                index = nxt
                continue
            data[key] = ""
            index += 1
            continue
        if value == "[]":
            data[key] = []
        else:
            data[key] = _unquote(value)
        index += 1
    return data, body


def dump_frontmatter(data: dict[str, object], keys: tuple[str, ...]) -> str:
    lines = ["---"]
    for key in keys:
        value = data.get(key, "")
        if isinstance(value, list):
            if not value:
                lines.append(key + ": []")
                continue
            lines.append(key + ":")
            for item in value:
                lines.append("  - " + json.dumps(str(item), ensure_ascii=False))
            continue
        lines.append(key + ": " + json.dumps("" if value is None else str(value), ensure_ascii=False))
    lines.append("---")
    return "\n".join(lines) + "\n"


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
        if isinstance(loaded, str):
            return loaded
    return value
