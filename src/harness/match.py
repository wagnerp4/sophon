from __future__ import annotations

import re
from pathlib import Path

_OPERATORS = ("&&", "||", "|&")
_TOKEN_HEAD = re.compile(r"^[^\s]+")


def parse_rule(rule: str) -> tuple[str, str | None]:
    text = str(rule or "").strip()
    if not text:
        return "", None
    if "(" not in text:
        return text, None
    if not text.endswith(")"):
        return text, None
    name, rest = text.split("(", 1)
    spec = rest[:-1].strip()
    if spec.endswith(":*") and ":" in spec:
        spec = spec[:-2].rstrip() + " *"
    if spec in ("", "*"):
        return name.strip(), None
    return name.strip(), spec


def format_rule(tool: str, specifier: str | None = None) -> str:
    name = str(tool or "").strip()
    if not specifier:
        return name
    return f"{name}({specifier})"


def split_compound(command: str) -> list[str]:
    raw = str(command or "")
    if not raw.strip():
        return []
    parts: list[str] = []
    buf: list[str] = []
    quote = ""
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            elif ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(raw[i + 1])
                i += 2
                continue
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        matched = False
        for op in _OPERATORS:
            if raw.startswith(op, i):
                piece = "".join(buf).strip()
                if piece:
                    parts.append(piece)
                buf = []
                i += len(op)
                matched = True
                break
        if matched:
            continue
        if ch in (";", "|", "&", "\n"):
            if ch == "&" and _is_redirection_amp(raw, i):
                buf.append(ch)
                i += 1
                continue
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    piece = "".join(buf).strip()
    if piece:
        parts.append(piece)
    return parts


def first_token(command: str) -> str:
    text = str(command or "").strip()
    if text.startswith("&"):
        text = text[1:].strip()
    lowered = text.lower()
    if lowered.startswith("sudo "):
        text = text[5:].strip()
    if not text:
        return ""
    if text[0] in "'\"":
        quote = text[0]
        end = text.find(quote, 1)
        raw = text[1:end] if end > 0 else text[1:]
    else:
        match = _TOKEN_HEAD.match(text)
        raw = match.group(0) if match else text
    raw = raw.strip().strip("'").strip('"')
    base = Path(raw).name
    if base.lower().endswith(".exe"):
        base = base[:-4]
    return base


def infer_persist_rules(
    tool: str,
    *,
    command: str = "",
    target: Path | None = None,
) -> list[str]:
    name = str(tool or "").strip()
    if not name:
        return []
    if name == "shell_exec":
        rules: list[str] = []
        seen: set[str] = set()
        parts = split_compound(command) or ([command.strip()] if command.strip() else [])
        for part in parts:
            token = first_token(part)
            if not token:
                continue
            rule = format_rule(name, f"{token} *")
            if rule in seen:
                continue
            seen.add(rule)
            rules.append(rule)
        return rules or [name]
    if target is not None:
        return [format_rule(name, str(target))]
    return [name]


def specifier_matches(pattern: str, text: str) -> bool:
    spec = str(pattern or "")
    body = str(text or "")
    if spec.endswith(":*"):
        spec = spec[:-2].rstrip() + " *"
    regex = _glob_to_re(spec)
    if regex.match(body):
        return True
    return regex.match(body.casefold()) is not None or regex.match(body.lower()) is not None


def rule_matches(
    rule: str,
    tool: str,
    *,
    command: str = "",
    target: Path | None = None,
) -> bool:
    name, spec = parse_rule(rule)
    if not name or name != tool:
        return False
    if spec is None:
        return True
    if command.strip():
        if specifier_matches(spec, command.strip()):
            return True
        return False
    if target is None:
        return False
    labels = (str(target), target.as_posix())
    try:
        labels = labels + (str(target.resolve()), target.resolve().as_posix())
    except OSError:
        pass
    return any(specifier_matches(spec, label) for label in labels)


def allow_command(rules: list[str], tool: str, command: str) -> bool:
    parts = split_compound(command) or ([command.strip()] if command.strip() else [])
    if not parts:
        return any(rule_matches(rule, tool, command=command) for rule in rules)
    for part in parts:
        if not any(rule_matches(rule, tool, command=part) for rule in rules):
            return False
    return True


def any_command_match(rules: list[str], tool: str, command: str) -> bool:
    if any(rule_matches(rule, tool, command=command) for rule in rules):
        return True
    for part in split_compound(command):
        if any(rule_matches(rule, tool, command=part) for rule in rules):
            return True
    return False


def _is_redirection_amp(text: str, index: int) -> bool:
    prev = text[index - 1] if index > 0 else ""
    nxt = text[index + 1] if index + 1 < len(text) else ""
    if prev == ">":
        return True
    if nxt == ">":
        return True
    if prev.isdigit() and nxt == ">":
        return True
    return False


def _glob_to_re(pattern: str) -> re.Pattern[str]:
    spec = str(pattern)
    if spec.endswith(" *"):
        prefix = spec[:-2]
        body = _escape_glob(prefix)
        return re.compile(r"^" + body + r"(?:\s.*)?$", re.IGNORECASE)
    return re.compile(r"^" + _escape_glob(spec) + r"$", re.IGNORECASE)


def _escape_glob(pattern: str) -> str:
    pieces = str(pattern).split("*")
    return ".*".join(re.escape(piece) for piece in pieces)
