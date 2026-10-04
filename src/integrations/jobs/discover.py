from __future__ import annotations

from integrations.jobs import paths
from integrations.jobs.notes import CompanySeed, company_path, load_company, render_company, slugify
from integrations.jobs.store import VaultStore


_DISCOVER_LIMIT = 10

_PROFILE_NAMES = (
    "Fernride",
    "Hive Robotics",
    "Tacterion",
    "OroraTech",
    "Munevo",
    "Konux",
    "Filics",
    "Holo-Light",
    "Aqarios",
    "planqc",
    "Reflex Aerospace",
    "The Exploration Company",
    "Xavveo",
    "Kaia Health",
)


def discover_jobs(store: VaultStore) -> str:
    snapshot = _ensure_snapshot(store)
    names = _snapshot_names(store.read(snapshot))
    known = _known_slugs(store)
    declined = _section_slugs(store, "Declined")
    pending = _pending_from_names(names, known, declined, snapshot, "On the Munich snapshot and not on the company list.")
    if len(pending) < _DISCOVER_LIMIT:
        extra = _search_names(known, declined, pending, _DISCOVER_LIMIT - len(pending))
        if extra:
            _append_names(store, snapshot, [row["legal_name"] for row in extra])
            pending.extend(extra)
    pending = pending[:_DISCOVER_LIMIT]
    store.write(paths.DISCOVER, _render(pending, declined))
    if not pending:
        return "discover: no new names for this profile"
    lines = [
        "discover: "
        + str(len(pending))
        + " to review. /jobs discover add N adds the first N."
    ]
    for index, row in enumerate(pending, start=1):
        lines.append(str(index) + ". " + row["legal_name"] + "  " + row["why"])
    return "\n".join(lines)


def discover_add(store: VaultStore, token: str) -> str:
    pending = _pending(store)
    chosen = _choose_add(pending, token)
    if not chosen:
        return "discover add: nothing pending for " + token
    left = list(pending)
    lines: list[str] = []
    added_names: list[str] = []
    for row in chosen:
        _write_added(store, row)
        left = [item for item in left if item["slug"] != row["slug"]]
        lines.append("added " + row["legal_name"] + " (" + row["slug"] + ")")
        lines.append(_probe_added(store, row["slug"]))
        added_names.append(row["legal_name"])
    store.write(paths.DISCOVER, _render(left, _section_slugs(store, "Declined")))
    from integrations.jobs.match import match_jobs

    matched = match_jobs(store)
    lines.append(matched)
    return "\n".join(lines)


def _probe_added(store: VaultStore, slug: str) -> str:
    from integrations.jobs.feeds import resolve_added

    try:
        return resolve_added(store, slug)
    except (OSError, RuntimeError, ValueError) as exc:
        return slug + ": probe failed (" + str(exc) + ")"


def discover_skip(store: VaultStore, token: str) -> str:
    pending = _pending(store)
    row = _pick(pending, token)
    if row is None:
        return "discover skip: no pending row for " + token
    left = [item for item in pending if item["slug"] != row["slug"]]
    declined = _section_slugs(store, "Declined")
    declined.add(row["slug"])
    store.write(paths.DISCOVER, _render(left, declined))
    return "skipped " + row["legal_name"]


def pending_count(store: VaultStore) -> int:
    if not store.exists(paths.DISCOVER):
        return 0
    return len(_pending(store))


def _latest_snapshot(store: VaultStore) -> str | None:
    names = [
        rel
        for rel in store.list_md(paths.SEEDTABLE)
        if rel.rsplit("/", 1)[-1].startswith("munich-ai-")
    ]
    if not names:
        return None
    return sorted(names)[-1]


def _ensure_snapshot(store: VaultStore) -> str:
    found = _latest_snapshot(store)
    if found is not None:
        return found
    from datetime import date

    rel = paths.SEEDTABLE + "/munich-ai-" + date.today().isoformat() + ".md"
    store.write(rel, "# Munich AI\n\n## Names\n")
    return rel


def _pending_from_names(
    names: list[str],
    known: set[str],
    declined: set[str],
    snapshot: str,
    why: str,
) -> list[dict[str, str]]:
    pending: list[dict[str, str]] = []
    seen: set[str] = set()
    for name in names:
        slug = slugify(name)
        if not slug or slug in known or slug in declined or slug in seen:
            continue
        seen.add(slug)
        pending.append(
            {
                "slug": slug,
                "legal_name": name,
                "why": why,
                "snapshot": snapshot,
            }
        )
    return pending


def _search_names(
    known: set[str],
    declined: set[str],
    pending: list[dict[str, str]],
    need: int,
) -> list[dict[str, str]]:
    taken = {row["slug"] for row in pending}
    found: list[str] = []
    try:
        from integrations.search.dispatch import search_web

        hits = search_web("Munich robotics AI startup perception autonomous", limit=10)
    except (OSError, RuntimeError, ValueError):
        hits = []
    for hit in hits:
        title = str(getattr(hit, "title", "") or "")
        name = title.split(" - ")[0].split(" | ")[0].split(" — ")[0].strip()
        if not name or len(name) > 40 or len(name.split()) > 4:
            continue
        found.append(name)
    found.extend(_PROFILE_NAMES)
    extra: list[dict[str, str]] = []
    for name in found:
        slug = slugify(name)
        if not slug or slug in known or slug in declined or slug in taken:
            continue
        taken.add(slug)
        extra.append(
            {
                "slug": slug,
                "legal_name": name,
                "why": "Profile search for Munich robotics, perception, AV, or neurotech.",
                "snapshot": "",
            }
        )
        if len(extra) >= need:
            break
    return extra


def _append_names(store: VaultStore, snapshot: str, names: list[str]) -> None:
    text = store.read(snapshot)
    if "## Names" not in text:
        text = text.rstrip() + "\n\n## Names\n"
    existing = {slugify(name) for name in _snapshot_names(text)}
    lines = [text.rstrip()]
    for name in names:
        if slugify(name) in existing:
            continue
        lines.append("- " + name)
        existing.add(slugify(name))
    store.write(snapshot, "\n".join(lines) + "\n")


def _choose_add(pending: list[dict[str, str]], token: str) -> list[dict[str, str]]:
    text = token.strip()
    if text.isdigit():
        count = int(text)
        if count < 1:
            return []
        return pending[:count]
    row = _pick_name(pending, text)
    if row is None:
        return []
    return [row]


def _write_added(store: VaultStore, row: dict[str, str]) -> None:
    rel = company_path(paths.COMPANIES, row["slug"])
    if not store.exists(rel):
        seed = CompanySeed(
            slug=row["slug"],
            legal_name=row["legal_name"],
            cluster="unclassified",
            note=row["why"],
            source="seedtable",
            munich="yes",
        )
        store.write(rel, render_company(seed, None))
        return
    text = store.read(rel)
    company = load_company(text, row["slug"])
    raw_aliases = company.get("aliases")
    aliases = [str(item) for item in raw_aliases] if isinstance(raw_aliases, list) else []
    seed = CompanySeed(
        slug=row["slug"],
        legal_name=str(company.get("legal_name") or row["legal_name"]),
        cluster=str(company.get("cluster") or "unclassified"),
        aliases=aliases,
        source="seedtable",
        munich=str(company.get("munich") or "yes"),
    )
    store.write(rel, render_company(seed, text))


def _pick_name(pending: list[dict[str, str]], token: str) -> dict[str, str] | None:
    slug = slugify(token)
    for row in pending:
        if row["slug"] == slug or slugify(row.get("legal_name", "")) == slug:
            return row
    return None


def _snapshot_names(text: str) -> list[str]:
    if "## Names" in text:
        text = text.split("## Names", 1)[1]
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if line.lower().startswith("source:") or line.lower().startswith("status:"):
            continue
        names.append(line)
    return names


def _known_slugs(store: VaultStore) -> set[str]:
    known: set[str] = set()
    for rel in store.list_md(paths.COMPANIES):
        slug = rel.rsplit("/", 1)[-1].removesuffix(".md")
        known.add(slug)
        if not store.exists(rel):
            continue
        company = load_company(store.read(rel), slug)
        legal = str(company.get("legal_name") or "")
        if legal:
            known.add(slugify(legal))
        aliases = company.get("aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                known.add(slugify(str(alias)))
    return known


def _pending(store: VaultStore) -> list[dict[str, str]]:
    if not store.exists(paths.DISCOVER):
        return []
    return _read_section(store.read(paths.DISCOVER), "Pending")


def _section_slugs(store: VaultStore, heading: str) -> set[str]:
    if not store.exists(paths.DISCOVER):
        return set()
    return {row["slug"] for row in _read_section(store.read(paths.DISCOVER), heading) if row.get("slug")}


def _read_section(text: str, heading: str) -> list[dict[str, str]]:
    marker = "## " + heading
    if marker not in text:
        return []
    block = text.split(marker, 1)[1]
    if "\n## " in block:
        block = block.split("\n## ", 1)[0]
    rows: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in block.splitlines():
        if raw.startswith("- slug:"):
            if current.get("slug"):
                rows.append(current)
            current = {"slug": raw.split(":", 1)[1].strip().strip("\"")}
            continue
        if raw.startswith("  ") and ":" in raw and current:
            key, value = raw.strip().split(":", 1)
            current[key.strip()] = value.strip().strip("\"")
    if current.get("slug"):
        rows.append(current)
    return rows


def _pick(pending: list[dict[str, str]], token: str) -> dict[str, str] | None:
    text = token.strip()
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(pending):
            return pending[index - 1]
        return None
    slug = slugify(text)
    for row in pending:
        if row["slug"] == slug or slugify(row.get("legal_name", "")) == slug:
            return row
    return None


def _render(pending: list[dict[str, str]], declined: set[str]) -> str:
    lines = ["# Discover", "", "## Pending", ""]
    for row in pending:
        lines.append("- slug: " + row["slug"])
        lines.append("  legal_name: " + row.get("legal_name", ""))
        lines.append("  why: " + row.get("why", ""))
        lines.append("  snapshot: " + row.get("snapshot", ""))
    lines.extend(["", "## Declined", ""])
    for slug in sorted(declined):
        lines.append("- slug: " + slug)
    lines.append("")
    return "\n".join(lines)
