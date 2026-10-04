from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date

from integrations.jobs import paths
from integrations.jobs.frontmatter import dump_frontmatter, parse_frontmatter
from integrations.jobs.notes import load_company
from integrations.jobs.store import VaultStore

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

def resolve_feeds(store: VaultStore, *, pause_s: float = 0.25) -> str:
    # TODO: record a human careers URL when every public ATS probe misses
    hits = 0
    checked = 0
    lines: list[str] = []
    for rel in store.list_md(paths.COMPANIES):
        text = store.read(rel)
        slug = rel.rsplit("/", 1)[-1].removesuffix(".md")
        company = load_company(text, slug)
        if str(company.get("cluster") or "") == "unclassified":
            continue
        if str(company.get("feed_url") or "").strip():
            continue
        checked += 1
        found = _probe(str(company.get("legal_name") or slug), slug, pause_s)
        if found is None:
            lines.append(slug + ": ats unknown")
            continue
        ats, feed_url, careers_url = found
        company["ats"] = ats
        company["feed_url"] = feed_url
        company["careers_url"] = careers_url
        _write_company(store, rel, company)
        hits += 1
        lines.append(slug + ": " + ats + " " + feed_url)
    if not lines:
        return "no classified companies to resolve"
    return "resolved " + str(hits) + " of " + str(checked) + "\n" + "\n".join(lines)


def fetch_roles(store: VaultStore, *, pause_s: float = 0.35) -> str:
    # TODO: HTML job-list fallback for ats html or unknown
    today = date.today().isoformat()
    updated = 0
    skipped = 0
    for rel in store.list_md(paths.COMPANIES):
        text = store.read(rel)
        slug = rel.rsplit("/", 1)[-1].removesuffix(".md")
        company = load_company(text, slug)
        if str(company.get("cluster") or "") == "unclassified":
            continue
        feed = str(company.get("feed_url") or "").strip()
        ats = str(company.get("ats") or "")
        if not feed or ats in ("", "unknown", "html", "workday"):
            skipped += 1
            continue
        status, body, etag = _http_get(feed, str(company.get("feed_etag") or ""))
        time.sleep(pause_s)
        if status == 304:
            company["last_fetched"] = today
            _write_company(store, rel, company)
            updated += 1
            continue
        if status != 200:
            continue
        roles = _parse_roles(ats, body, feed)
        company["body"] = _merge_roles(str(company.get("body") or ""), roles, today)
        company["last_fetched"] = today
        if etag:
            company["feed_etag"] = etag
        _write_company(store, rel, company)
        updated += 1
    return "fetched roles for " + str(updated) + " companies, skipped " + str(skipped)


_HINTS = {
    "agile-robots": ["agilerobotsse"],
    "anima-machina": ["animamachina"],
    "manex-ai": ["manexai", "manex"],
    "nomadic-drones": ["nomadicdrones"],
    "36zero-vision": ["36zerovision"],
    "se3labs": ["se3"],
}

# TODO: Recruitee is outside Greenhouse, Lever, Ashby, and Personio. Keep it only where the public offers JSON is known.
_DIRECT = {
    "manex-ai": (
        "ashby",
        "https://api.ashbyhq.com/posting-api/job-board/manex",
        "https://jobs.ashbyhq.com/manex",
    ),
    "se3labs": (
        "ashby",
        "https://api.ashbyhq.com/posting-api/job-board/se3",
        "https://jobs.ashbyhq.com/se3",
    ),
    "tytan": (
        "recruitee",
        "https://tytantechnologiesgmbh.recruitee.com/api/offers",
        "https://tytantechnologiesgmbh.recruitee.com/",
    ),
}


def _probe(legal: str, slug: str, pause_s: float) -> tuple[str, str, str] | None:
    tokens: list[str] = []
    for token in [slug, slug.replace("-", ""), re.sub(r"[^a-z0-9]", "", legal.lower()), *_HINTS.get(slug, [])]:
        if token and token not in tokens:
            tokens.append(token)
    tokens = tokens[:4]
    empty: tuple[str, str, str] | None = None
    direct = _DIRECT.get(slug)
    if direct is not None:
        ats, url, careers = direct
        status, body, _etag = _http_get(url, "")
        time.sleep(pause_s)
        roles = _parse_roles(ats, body, url) if status == 200 else None
        if roles:
            return ats, url, careers
        if roles is not None:
            empty = (ats, url, careers)
    for token in tokens:
        personio = "https://" + token + ".jobs.personio.de/xml"
        greenhouse = "https://boards-api.greenhouse.io/v1/boards/" + token + "/jobs"
        lever = "https://api.lever.co/v0/postings/" + token + "?mode=json"
        ashby = "https://api.ashbyhq.com/posting-api/job-board/" + token
        trials = (
            ("personio", personio, "https://" + token + ".jobs.personio.de/"),
            ("greenhouse", greenhouse, "https://boards.greenhouse.io/" + token),
            ("lever", lever, "https://jobs.lever.co/" + token),
            ("ashby", ashby, "https://jobs.ashbyhq.com/" + token),
        )
        for ats, url, careers in trials:
            status, body, _etag = _http_get(url, "")
            if status == 429:
                time.sleep(max(pause_s, 1.5))
                status, body, _etag = _http_get(url, "")
            time.sleep(pause_s)
            if status != 200:
                continue
            roles = _parse_roles(ats, body, url)
            if roles is None:
                continue
            found = (ats, url, careers)
            if roles:
                return found
            if empty is None:
                empty = found
    return empty


def _parse_roles(ats: str, body: str, feed_url: str) -> list[dict[str, str]] | None:
    lowered = body.lstrip().lower()
    if lowered.startswith("<!doctype html") or lowered.startswith("<html"):
        return None
    if ats == "personio":
        return _parse_personio(body, feed_url)
    if ats == "lever":
        return _parse_lever(body)
    if ats == "recruitee":
        return _parse_recruitee(body)
    if ats in ("greenhouse", "ashby"):
        return _parse_job_objects(body)
    return None


def _parse_personio(body: str, feed_url: str) -> list[dict[str, str]] | None:
    if "<position" not in body.lower() and "workzag" not in body.lower():
        return None
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None
    host = feed_url.split("/xml")[0].rstrip("/")
    roles: list[dict[str, str]] = []
    for node in root.iter():
        if _local(node.tag) != "position":
            continue
        title = _child_text(node, "name")
        if not title:
            continue
        job_id = _child_text(node, "id")
        url = host + "/job/" + job_id if job_id else ""
        roles.append(
            {
                "title": title,
                "url": url,
                "location": _child_text(node, "office"),
                "team": _child_text(node, "department"),
            }
        )
    return roles


def _parse_recruitee(body: str) -> list[dict[str, str]] | None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    offers = data.get("offers") if isinstance(data, dict) else None
    if not isinstance(offers, list):
        return None
    roles: list[dict[str, str]] = []
    for item in offers:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        if not title:
            continue
        roles.append(
            {
                "title": title,
                "url": str(item.get("careers_url") or ""),
                "location": str(item.get("location") or item.get("city") or ""),
                "team": str(item.get("department") or ""),
            }
        )
    return roles


def _parse_lever(body: str) -> list[dict[str, str]] | None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    roles: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        categories = item.get("categories") if isinstance(item.get("categories"), dict) else {}
        roles.append(
            {
                "title": str(item.get("text") or ""),
                "url": str(item.get("hostedUrl") or ""),
                "location": str(categories.get("location") or ""),
                "team": str(categories.get("team") or ""),
            }
        )
    return [role for role in roles if role["title"]]


def _parse_job_objects(body: str) -> list[dict[str, str]] | None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, list):
        return None
    roles: list[dict[str, str]] = []
    for item in jobs:
        if not isinstance(item, dict):
            continue
        location = item.get("location")
        if isinstance(location, dict):
            place = str(location.get("name") or "")
        else:
            place = str(location or "")
        title = str(item.get("title") or item.get("name") or "")
        url = str(item.get("absolute_url") or item.get("jobUrl") or item.get("applyUrl") or "")
        team = str(item.get("department") or item.get("team") or "")
        if title:
            roles.append({"title": title, "url": url, "location": place, "team": team})
    return roles


def _merge_roles(body: str, roles: list[dict[str, str]], today: str) -> str:
    existing = _read_roles(body)
    by_key: dict[str, dict[str, str]] = {}
    for role in existing:
        by_key[_role_key(role)] = role
    seen: set[str] = set()
    for role in roles:
        key = _role_key(role)
        seen.add(key)
        prior = by_key.get(key)
        first = str(prior.get("first_seen") or today) if prior else today
        by_key[key] = {
            "title": role.get("title") or (prior or {}).get("title", ""),
            "url": role.get("url") or (prior or {}).get("url", ""),
            "location": role.get("location") or (prior or {}).get("location", ""),
            "team": role.get("team") or (prior or {}).get("team", ""),
            "first_seen": first,
            "last_seen": today,
            "status": _kept_status(prior, role),
            "seen_count": _next_seen(prior),
        }
    for key, role in by_key.items():
        if key not in seen:
            role.setdefault("first_seen", today)
            role.setdefault("last_seen", role.get("last_seen") or today)
            role["status"] = _kept_status(role)
            role["seen_count"] = str(role.get("seen_count") or "1")
    head = body
    marker = "## Roles"
    if marker in body:
        head = body.split(marker, 1)[0].rstrip()
    lines = [head, "", "## Roles", ""]
    for role in by_key.values():
        if not role.get("title"):
            continue
        lines.append("- title: " + json.dumps(role.get("title", ""), ensure_ascii=False))
        lines.append("  url: " + json.dumps(role.get("url", ""), ensure_ascii=False))
        lines.append("  location: " + json.dumps(role.get("location", ""), ensure_ascii=False))
        lines.append("  team: " + json.dumps(role.get("team", ""), ensure_ascii=False))
        lines.append("  first_seen: " + json.dumps(role.get("first_seen", today), ensure_ascii=False))
        lines.append("  last_seen: " + json.dumps(role.get("last_seen", today), ensure_ascii=False))
        lines.append("  status: " + json.dumps(role.get("status") or "open", ensure_ascii=False))
        lines.append("  seen_count: " + str(_seen_int(role.get("seen_count"))))
    return "\n".join(lines).rstrip() + "\n"


def set_role_status(store: VaultStore, slug: str, url: str, status: str, title: str = "") -> str:
    if status not in ("open", "applied", "skip"):
        raise ValueError("status must be open, applied, or skip")
    rel, company, roles = _company_roles(store, slug)
    changed = 0
    for role in roles:
        same_url = bool(url) and role.get("url") == url
        same_title = bool(title) and not url and role.get("title") == title
        if same_url or same_title:
            role["status"] = status
            changed += 1
    if changed == 0:
        raise ValueError("role not found")
    _save_roles(store, rel, company, roles)
    return status


def set_company_applied(store: VaultStore, slug: str) -> int:
    rel, company, roles = _company_roles(store, slug)
    changed = 0
    for role in roles:
        if (role.get("status") or "open") == "open":
            role["status"] = "applied"
            changed += 1
    _save_roles(store, rel, company, roles)
    return changed


def resolve_added(store: VaultStore, slug: str, *, pause_s: float = 0.3) -> str:
    rel = paths.COMPANIES + "/" + slug + ".md"
    if not store.exists(rel):
        return "no company note for " + slug
    company = load_company(store.read(rel), slug)
    feed = str(company.get("feed_url") or "").strip()
    ats = str(company.get("ats") or "")
    if not feed or ats in ("", "unknown", "html", "workday"):
        found = _probe(str(company.get("legal_name") or slug), slug, pause_s)
        if found is None:
            return slug + ": ats unknown"
        ats, feed, careers = found
        company["ats"] = ats
        company["feed_url"] = feed
        company["careers_url"] = careers
    status, body, etag = _http_get(feed, str(company.get("feed_etag") or ""))
    if status != 200:
        _write_company(store, rel, company)
        return slug + ": feed " + feed + " status " + str(status)
    parsed = _parse_roles(ats, body, feed) or []
    today = date.today().isoformat()
    company["body"] = _merge_roles(str(company.get("body") or ""), parsed, today)
    company["last_fetched"] = today
    if etag:
        company["feed_etag"] = etag
    _write_company(store, rel, company)
    return slug + ": " + ats + " " + str(len(parsed)) + " roles"


def _company_roles(store: VaultStore, slug: str) -> tuple[str, dict[str, object], list[dict[str, str]]]:
    rel = paths.COMPANIES + "/" + slug + ".md"
    if not store.exists(rel):
        raise ValueError("no company note for " + slug)
    company = load_company(store.read(rel), slug)
    roles = _read_roles(str(company.get("body") or ""))
    return rel, company, roles


def _save_roles(store: VaultStore, rel: str, company: dict[str, object], roles: list[dict[str, str]]) -> None:
    today = date.today().isoformat()
    body = str(company.get("body") or "")
    head = body.split("## Roles", 1)[0].rstrip() if "## Roles" in body else body.rstrip()
    lines = [head, "", "## Roles", ""]
    for role in roles:
        if not role.get("title"):
            continue
        lines.append("- title: " + json.dumps(role.get("title", ""), ensure_ascii=False))
        lines.append("  url: " + json.dumps(role.get("url", ""), ensure_ascii=False))
        lines.append("  location: " + json.dumps(role.get("location", ""), ensure_ascii=False))
        lines.append("  team: " + json.dumps(role.get("team", ""), ensure_ascii=False))
        lines.append("  first_seen: " + json.dumps(role.get("first_seen") or today, ensure_ascii=False))
        lines.append("  last_seen: " + json.dumps(role.get("last_seen") or today, ensure_ascii=False))
        lines.append("  status: " + json.dumps(_kept_status(role), ensure_ascii=False))
        lines.append("  seen_count: " + str(_seen_int(role.get("seen_count") or "1")))
    company["body"] = "\n".join(lines).rstrip() + "\n"
    _write_company(store, rel, company)


def _kept_status(prior: dict[str, str] | None, incoming: dict[str, str] | None = None) -> str:
    for source in (prior, incoming):
        status = str((source or {}).get("status") or "")
        if status in ("applied", "skip"):
            return status
    return "open"


def _next_seen(prior: dict[str, str] | None) -> str:
    return str(_seen_int((prior or {}).get("seen_count")) + 1)


def _seen_int(value: object) -> int:
    try:
        return max(0, int(str(value or "0").strip() or "0"))
    except ValueError:
        return 0


def read_roles(body: str) -> list[dict[str, str]]:
    return _read_roles(body)


def _read_roles(body: str) -> list[dict[str, str]]:
    if "## Roles" in body:
        body = body.split("## Roles", 1)[1]
    roles: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in body.splitlines():
        if raw.startswith("- title:"):
            if current.get("title"):
                roles.append(current)
            current = {"title": _unquote_field(raw.split(":", 1)[1])}
            continue
        if raw.startswith("  ") and ":" in raw and current:
            key, value = raw.strip().split(":", 1)
            current[key.strip()] = _unquote_field(value)
    if current.get("title"):
        roles.append(current)
    return roles


def _write_company(store: VaultStore, rel: str, company: dict[str, object]) -> None:
    data, body = parse_frontmatter(store.read(rel))
    for key in _FRONT_KEYS:
        if key == "aliases":
            aliases = company.get("aliases")
            data[key] = aliases if isinstance(aliases, list) else data.get(key, [])
            continue
        if key in company and key != "body":
            data[key] = company.get(key, "")
    new_body = company.get("body")
    if isinstance(new_body, str) and new_body.strip():
        body = new_body
    store.write(rel, dump_frontmatter(data, _FRONT_KEYS) + "\n" + body.lstrip("\n"))


def _http_get(url: str, etag: str) -> tuple[int, str, str]:
    headers = {
        "User-Agent": "sophon-jobs/0.1",
        "Accept": "application/json, application/xml, text/xml, */*",
    }
    if etag:
        headers["If-None-Match"] = etag
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), payload, str(resp.headers.get("ETag") or "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), detail[:4000], ""
    except (OSError, urllib.error.URLError):
        return 0, "", ""


def _child_text(node: ET.Element, name: str) -> str:
    for child in list(node):
        if _local(child.tag) == name:
            return "".join(child.itertext()).strip()
    return ""


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _role_key(role: dict[str, str]) -> str:
    url = role.get("url") or ""
    if url:
        return url
    return (role.get("title") or "") + "|" + (role.get("location") or "")


def _unquote_field(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == "\"":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return text.strip("\"")
        if isinstance(loaded, str):
            return loaded
    return text
