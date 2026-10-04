from __future__ import annotations

import re
from dataclasses import dataclass, field

from integrations.jobs.frontmatter import dump_frontmatter, parse_frontmatter

_FRONT_KEYS = (
    "legal_name",
    "cluster",
    "source",
    "hq",
    "munich",
    "careers_url",
    "ats",
    "feed_url",
    "feed_etag",
    "last_fetched",
    "fit",
    "aliases",
)

_MUNICH_WORDS = (
    "munich",
    "münchen",
    "muenchen",
    "gilching",
    "garching",
    "unterschleißheim",
    "unterschleissheim",
    "martinsried",
    "oberpfaffenhofen",
)

_SLUG_FIX = {
    "agile-robotics": "agile-robots",
    "agile-robots-se": "agile-robots",
}

_LEGAL_FIX = {
    "agile-robots": "Agile Robots SE",
}

_SPLIT = {
    frozenset({"silabs", "infineon"}),
    frozenset({"fraunhofer-ipa", "fraunhofer-iks"}),
}

_MERGE = {
    frozenset({"autonomous-intelligent-driving-aid", "cariad"}): (
        "cariad",
        "CARIAD",
        ["Autonomous Intelligent Driving", "AID"],
    ),
    frozenset({"customcells", "automotive-ai-startups"}): ("customcells", "Customcells", []),
    frozenset({"oura", "wearable-neurotech-startups"}): ("oura", "Oura", []),
    frozenset({"unternehmertum", "makerspace"}): ("unternehmertum", "UnternehmerTUM", ["MakerSpace"]),
}


@dataclass
class CompanySeed:
    slug: str
    legal_name: str
    cluster: str
    aliases: list[str] = field(default_factory=list)
    note: str = ""
    munich: str = ""
    source: str = "vault"


def slugify(name: str) -> str:
    text = name.strip().lower().replace("&", " and ")
    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("-")
    slug = "".join(chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    return _SLUG_FIX.get(slug, slug)


def parse_classified(text: str) -> list[CompanySeed]:
    cluster = ""
    seeds: list[CompanySeed] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            cluster = _cluster_heading(line)
            continue
        if not cluster:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        line = line.strip("*").strip()
        if not line:
            continue
        seeds.append(
            CompanySeed(
                slug=slugify(line),
                legal_name=_LEGAL_FIX.get(slugify(line), line),
                cluster=cluster,
                aliases=[line] if line != _LEGAL_FIX.get(slugify(line), line) else [],
                munich="yes",
            )
        )
    return _collapse(seeds)


def parse_unclassified(text: str) -> list[CompanySeed]:
    seeds: list[CompanySeed] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("-"):
            continue
        bolds = re.findall(r"\*\*(.+?)\*\*", line)
        note = _paren_note(line)
        munich = "yes" if any(word in line.lower() for word in _MUNICH_WORDS) else ""
        if not bolds:
            name = line.lstrip("-").strip()
            if name:
                seeds.append(_one(name, note, munich))
            continue
        keys = frozenset(slugify(part) for part in bolds)
        if keys in _MERGE:
            slug, legal, aliases = _MERGE[keys]
            seeds.append(
                CompanySeed(
                    slug=slug,
                    legal_name=legal,
                    cluster="unclassified",
                    aliases=aliases,
                    note=note,
                    munich=munich,
                )
            )
            continue
        if keys in _SPLIT or len(bolds) > 1 and keys not in _MERGE:
            if keys in _SPLIT:
                for part in bolds:
                    seeds.append(_one(part, note, munich))
                continue
        seeds.append(_one(bolds[0], note, munich))
    return _collapse(seeds)


def render_company(seed: CompanySeed, existing: str | None) -> str:
    if existing:
        data, body = parse_frontmatter(existing)
        data["legal_name"] = seed.legal_name or str(data.get("legal_name") or "")
        old_cluster = str(data.get("cluster") or "")
        if seed.cluster != "unclassified" and (not old_cluster or old_cluster == "unclassified"):
            data["cluster"] = seed.cluster
        elif not old_cluster:
            data["cluster"] = seed.cluster
        data["source"] = _merge_source(str(data.get("source") or ""), seed.source)
        if not str(data.get("hq") or ""):
            data["hq"] = ""
        old_munich = str(data.get("munich") or "")
        if seed.munich == "yes" or not old_munich:
            data["munich"] = seed.munich or old_munich
        for key in ("careers_url", "ats", "feed_url", "feed_etag", "last_fetched", "fit"):
            data.setdefault(key, "")
        if not str(data.get("ats") or ""):
            data["ats"] = "unknown"
        aliases = data.get("aliases")
        merged = []
        if isinstance(aliases, list):
            merged.extend(str(item) for item in aliases)
        merged.extend(seed.aliases)
        data["aliases"] = _unique(merged)
        if "## Roles" not in body:
            body = body.rstrip() + "\n\n## Roles\n"
        return dump_frontmatter(data, _FRONT_KEYS) + "\n" + body.lstrip("\n")
    data = {
        "legal_name": seed.legal_name,
        "cluster": seed.cluster,
        "source": seed.source,
        "hq": "",
        "munich": seed.munich,
        "careers_url": "",
        "ats": "unknown",
        "feed_url": "",
        "feed_etag": "",
        "last_fetched": "",
        "fit": "",
        "aliases": _unique(seed.aliases),
    }
    if seed.cluster == "unclassified":
        where = "On the unclassified list."
    else:
        where = "On the classified list under " + seed.cluster + "."
    body_lines = [where]
    if seed.note:
        body_lines.extend(["", seed.note])
    body_lines.extend(["", "## Roles", ""])
    return dump_frontmatter(data, _FRONT_KEYS) + "\n" + "\n".join(body_lines)


def company_path(folder: str, slug: str) -> str:
    return folder.rstrip("/") + "/" + slug + ".md"


def load_company(text: str, slug: str) -> dict[str, object]:
    data, body = parse_frontmatter(text)
    data["slug"] = slug
    data["body"] = body
    aliases = data.get("aliases")
    if not isinstance(aliases, list):
        data["aliases"] = []
    return data


def _one(name: str, note: str, munich: str) -> CompanySeed:
    slug = slugify(name)
    return CompanySeed(
        slug=slug,
        legal_name=_LEGAL_FIX.get(slug, name.strip()),
        cluster="unclassified",
        aliases=[],
        note=note,
        munich=munich,
    )


def _collapse(seeds: list[CompanySeed]) -> list[CompanySeed]:
    by_slug: dict[str, CompanySeed] = {}
    for seed in seeds:
        if not seed.slug:
            continue
        current = by_slug.get(seed.slug)
        if current is None:
            by_slug[seed.slug] = seed
            continue
        if seed.cluster != "unclassified":
            current.cluster = seed.cluster
            current.munich = "yes"
        if seed.legal_name and seed.cluster != "unclassified":
            current.legal_name = seed.legal_name
        if seed.note and seed.note not in current.note:
            current.note = (current.note + " " + seed.note).strip()
        for alias in seed.aliases:
            if alias not in current.aliases and alias != current.legal_name:
                current.aliases.append(alias)
        if seed.munich == "yes":
            current.munich = "yes"
    return list(by_slug.values())


def _cluster_heading(line: str) -> str:
    low = line.lower()
    if "embodied" in low:
        return "embodied"
    if "perception" in low:
        return "perception"
    if "quality" in low or "industrial" in low:
        return "industrial"
    if "neuro" in low:
        return "neuroscience"
    if re.search(r"\bav\b", low):
        return "av"
    return ""


def _paren_note(line: str) -> str:
    match = re.search(r"\)\s*\((.+)\)\s*$", line)
    if match and "**" not in match.group(1):
        return match.group(1).strip()
    match = re.search(r"\*\*\s*\((.+)\)\s*$", line)
    if match:
        return match.group(1).strip()
    return ""


def _merge_source(old: str, new: str) -> str:
    parts: set[str] = set()
    for chunk in (old, new):
        for bit in chunk.replace("both", "vault,seedtable").split(","):
            bit = bit.strip()
            if bit:
                parts.add(bit)
    if "vault" in parts and ("seedtable" in parts or "discover" in parts):
        return "both"
    if not parts:
        return "vault"
    if len(parts) == 1:
        return next(iter(parts))
    return "both"


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = item.strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
