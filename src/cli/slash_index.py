from __future__ import annotations

from dataclasses import dataclass

from cli.chat import list_slash_commands
from cli.chat_display import ESSENTIAL_SLASH_COMMANDS

EMPTY_QUERY_LIMIT = 8
RECENTS_CAP = 16

_SECTION_ORDER: tuple[str, ...] = (
    "session",
    "model",
    "generation",
    "tools",
    "subagents",
    "speech",
    "memory",
    "skills",
    "eval",
    "train",
    "other",
)


@dataclass(frozen=True)
class SlashEntry:
    name: str
    help_text: str
    section: str
    aliases: tuple[str, ...]
    shortcut: str
    section_index: int


_CATALOG: tuple[SlashEntry, ...] | None = None
_ESSENTIAL_NAMES: tuple[str, ...] | None = None


def remember_slash_use(recents: list[str], name: str, *, cap: int = RECENTS_CAP) -> list[str]:
    canonical = str(name or "").strip().lstrip("/").lower()
    if not canonical:
        return list(recents)
    out = [canonical]
    for item in recents:
        token = str(item or "").strip().lstrip("/").lower()
        if token and token != canonical:
            out.append(token)
    return out[:cap]


def _essential_names() -> tuple[str, ...]:
    global _ESSENTIAL_NAMES
    if _ESSENTIAL_NAMES is not None:
        return _ESSENTIAL_NAMES
    names: list[str] = []
    seen: set[str] = set()
    for raw, _help in ESSENTIAL_SLASH_COMMANDS:
        token = raw.strip().split()[0].lstrip("/").lower()
        if not token or token in seen:
            continue
        seen.add(token)
        names.append(token)
    _ESSENTIAL_NAMES = tuple(names)
    return _ESSENTIAL_NAMES


def _section_index(section: str) -> int:
    try:
        return _SECTION_ORDER.index(section)
    except ValueError:
        return len(_SECTION_ORDER)


def slash_catalog() -> tuple[SlashEntry, ...]:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    rows: list[SlashEntry] = []
    for info in list_slash_commands():
        rows.append(
            SlashEntry(
                name=info.name,
                help_text=info.help_text,
                section=info.section,
                aliases=info.aliases,
                shortcut=info.shortcut,
                section_index=_section_index(info.section),
            )
        )
    _CATALOG = tuple(rows)
    return _CATALOG


def slash_query_from_value(value: str) -> str | None:
    text = value or ""
    if not text.startswith("/"):
        return None
    rest = text[1:]
    if " " in rest or "\t" in rest:
        return None
    return rest


def _match_rank(entry: SlashEntry, query: str, recents: list[str]) -> tuple[int, int, int, int, int, int, str] | None:
    q = query.lower()
    name = entry.name.lower()
    aliases = tuple(alias.lower() for alias in entry.aliases)
    recent_idx = recents.index(name) if name in recents else len(recents) + 1
    prefix_name = 0 if (not q or name.startswith(q)) else 1
    prefix_alias = 0 if q and any(alias.startswith(q) for alias in aliases) else 1
    substr_name = 0 if (not q or q in name) else 1
    substr_help = 0 if q and q in entry.help_text.lower() else 1
    if q:
        matched = prefix_name == 0 or prefix_alias == 0 or substr_name == 0 or substr_help == 0
        if not matched:
            return None
        if prefix_name != 0 and prefix_alias != 0:
            recent_idx = len(recents) + 1
    essential = _essential_names()
    essential_idx = essential.index(name) if name in essential else len(essential) + 1
    return (
        recent_idx if prefix_name == 0 or not q else len(recents) + 1,
        prefix_name,
        prefix_alias,
        substr_name,
        essential_idx if not q else 0,
        entry.section_index,
        name,
    )


def rank_slash_commands(
    query: str,
    recents: list[str] | None = None,
    *,
    limit: int = EMPTY_QUERY_LIMIT,
) -> list[SlashEntry]:
    q = (query or "").strip().lstrip("/").lower()
    recent_names: list[str] = []
    seen: set[str] = set()
    for item in recents or []:
        token = str(item or "").strip().lstrip("/").lower()
        if not token or token in seen:
            continue
        seen.add(token)
        recent_names.append(token)
    catalog = slash_catalog()
    ranked: list[tuple[tuple, SlashEntry]] = []
    for entry in catalog:
        key = _match_rank(entry, q, recent_names)
        if key is None:
            continue
        ranked.append((key, entry))
    ranked.sort(key=lambda item: item[0])
    hits = [entry for _key, entry in ranked]
    if limit > 0:
        return hits[:limit]
    return hits


def format_slash_row(entry: SlashEntry, width: int = 72) -> str:
    help_text = " ".join((entry.help_text or "").split())
    label = "/" + entry.name
    if entry.shortcut:
        label = label + "  " + entry.shortcut
    room = max(12, width - len(label) - 3)
    if len(help_text) > room:
        help_text = help_text[: max(0, room - 1)] + "…"
    if help_text:
        return f"{label}  {help_text}"
    return label
