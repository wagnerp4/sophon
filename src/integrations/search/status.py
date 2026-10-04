from __future__ import annotations

from integrations.google.searxng import searxng_base_url, searxng_configured
from integrations.search.cse import cse_configured
from integrations.search.dispatch import (
    clear_disabled_sources,
    disabled_names,
    search_to_markdown,
    search_web,
    set_source_disabled,
    set_web_search_tools,
    web_search_tools_enabled,
)
from integrations.search.http import search_contact, set_search_contact
from integrations.search.registry import get, names

SEARCH_COMMAND_USAGE = (
    "usage: /search | on | off | contact EMAIL|clear | "
    "enable SOURCE|all | disable SOURCE | SOURCE [QUERY]"
)

_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("web", ("auto", "searxng", "cse")),
    ("knowledge", ("wikipedia", "wikidata", "ddg")),
    ("academic", ("arxiv", "crossref", "openalex", "pubmed", "europepmc", "semanticscholar", "zenodo")),
    ("code", ("github", "huggingface", "stackexchange", "hn", "mdn")),
)

_SUMMARIES: dict[str, str] = {
    "auto": "Broad web. SearXNG, then CSE. Does not fan out to verticals.",
    "searxng": "Self-hosted metasearch. JSON must be enabled. SOPHON_SEARXNG_URL.",
    "cse": "Google Custom Search. SOPHON_GOOGLE_CSE_KEY and SOPHON_GOOGLE_CSE_CX.",
    "wikipedia": "OpenSearch plus a short extract. extras.license=CC BY-SA.",
    "wikidata": "wbsearchentities keyword search. Not SPARQL.",
    "ddg": "DuckDuckGo Instant Answer JSON. Often empty. Not a web SERP.",
    "arxiv": "Atom API. 3s minimum interval.",
    "crossref": "Works query. mailto when SOPHON_SEARCH_CONTACT is set.",
    "openalex": "Works search. mailto when SOPHON_SEARCH_CONTACT is set.",
    "pubmed": "E-utilities esearch then esummary. tool=sophon.",
    "europepmc": "Europe PMC REST format=json.",
    "semanticscholar": "Graph paper/search. Unauthenticated pool.",
    "github": "Repository search. Unauthenticated quota is low.",
    "huggingface": "Public model and dataset search.",
    "zenodo": "Open research records and datasets. mostrecent sort.",
    "stackexchange": "Stack Overflow advanced search. 300/day without a key.",
    "hn": "Hacker News Algolia.",
    "mdn": "MDN document search.",
}


def family_of(name: str) -> str:
    key = name.strip().lower()
    for family, members in _FAMILIES:
        if key in members:
            return family
    return "other"


def _ready_label(name: str, *, blocked: frozenset[str]) -> str:
    if name == "auto":
        if blocked.intersection({"searxng", "cse"}) == {"searxng", "cse"}:
            return "disabled"
        if searxng_configured() and "searxng" not in blocked:
            return "ready"
        if cse_configured() and "cse" not in blocked:
            return "ready"
        return "missing"
    if name in blocked:
        return "disabled"
    source = get(name)
    return "ready" if source.available() else "missing"


def _note(name: str, *, blocked: frozenset[str]) -> str:
    if name == "auto":
        bits: list[str] = []
        if searxng_configured() and "searxng" not in blocked:
            bits.append(f"searxng {searxng_base_url()}")
        elif "searxng" in blocked:
            bits.append("searxng disabled")
        else:
            bits.append("searxng missing")
        if cse_configured() and "cse" not in blocked:
            bits.append("then cse")
        elif "cse" in blocked:
            bits.append("cse disabled")
        else:
            bits.append("cse missing")
        return " ".join(bits)
    if name == "searxng":
        url = searxng_base_url()
        return url if url else "SOPHON_SEARXNG_URL unset"
    if name == "cse":
        return "CSE key+cx set" if cse_configured() else "SOPHON_GOOGLE_CSE_KEY / CX unset"
    source = get(name)
    if source.min_interval_s > 0:
        return f"{source.min_interval_s:g}s interval"
    return _SUMMARIES.get(name, "")


def format_search_status() -> str:
    blocked = disabled_names()
    contact = search_contact() or "(empty)"
    tool = "on" if web_search_tools_enabled() else "off"
    lines = [
        f"web_search tool: {tool}  (SOPHON_WEB_SEARCH_TOOLS, /search on|off)",
        f"contact: {contact}  (SOPHON_SEARCH_CONTACT, /search contact EMAIL|clear)",
        (
            "auto: SearXNG then CSE. Named source= calls one backend. "
            "google off is Gmail/Drive."
        ),
        SEARCH_COMMAND_USAGE,
        "",
        f"{'source':<16} {'family':<10} {'state':<8} note",
    ]
    for family, members in _FAMILIES:
        for name in members:
            if name != "auto":
                get(name)
            state = _ready_label(name, blocked=blocked)
            note = _note(name, blocked=blocked)
            lines.append(f"{name:<16} {family:<10} {state:<8} {note}")
    extra = [name for name in names() if family_of(name) == "other"]
    for name in extra:
        state = _ready_label(name, blocked=blocked)
        note = _note(name, blocked=blocked)
        lines.append(f"{name:<16} {'other':<10} {state:<8} {note}")
    if blocked:
        lines.append("")
        lines.append("disabled: " + ", ".join(sorted(blocked)))
    return "\n".join(lines)


def format_search_source(name: str) -> str:
    key = name.strip().lower() or "auto"
    blocked = disabled_names()
    if key != "auto":
        source = get(key)
        interval = source.min_interval_s
        available = source.available()
    else:
        interval = 0.0
        available = _ready_label("auto", blocked=blocked) == "ready"
    lines = [
        f"source: {key}",
        f"family: {family_of(key)}",
        f"state: {_ready_label(key, blocked=blocked)}",
        f"available: {'yes' if available else 'no'}",
        f"disabled: {'yes' if key in blocked else 'no'}",
        f"interval_s: {interval:g}",
        f"summary: {_SUMMARIES.get(key, '')}",
        f"note: {_note(key, blocked=blocked)}",
    ]
    return "\n".join(lines)


def handle_search_command(arg: str) -> str:
    text = (arg or "").strip()
    if not text or text.lower() in {"show", "list", "status"}:
        return format_search_status()
    parts = text.split(None, 1)
    verb = parts[0].strip().lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if verb in {"on", "1", "true", "yes"}:
        set_web_search_tools(True)
        return "web_search tool: on"
    if verb in {"off", "0", "false", "no"}:
        set_web_search_tools(False)
        return "web_search tool: off"
    if verb == "help":
        return SEARCH_COMMAND_USAGE
    if verb == "contact":
        if not rest:
            current = search_contact() or "(empty)"
            return f"contact: {current}"
        value = set_search_contact(rest)
        return f"contact: {value or '(empty)'}"
    if verb == "disable":
        if not rest:
            return "usage: /search disable SOURCE"
        set_source_disabled(rest, disabled=True)
        return f"disabled: {rest.strip().lower()}"
    if verb == "enable":
        if not rest:
            return "usage: /search enable SOURCE|all"
        if rest.strip().lower() == "all":
            clear_disabled_sources()
            return "disabled: (none)"
        set_source_disabled(rest, disabled=False)
        return f"enabled: {rest.strip().lower()}"
    registered = set(names())
    registered.add("auto")
    if verb in registered:
        if not rest:
            return format_search_source(verb)
        hits = search_web(rest, limit=5, source=verb)
        return search_to_markdown(hits, query=rest)
    return SEARCH_COMMAND_USAGE
