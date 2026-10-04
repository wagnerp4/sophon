from __future__ import annotations

import re
from datetime import date

from integrations.jobs import paths
from integrations.jobs.criterion import load_criterion
from integrations.jobs.feeds import read_roles
from integrations.jobs.ingest import read_profile_path
from integrations.jobs.notes import load_company
from integrations.jobs.profile import redact_profile
from integrations.jobs.store import VaultStore

_BIG = {
    "apple",
    "google",
    "meta",
    "bmw-group",
    "bmw",
    "intel",
    "qualcomm",
    "infineon",
    "kuka",
    "celonis",
}

_CLUSTER_NEEDLES = {
    "embodied": ("robot", "cognitive", "manipulation", "ics"),
    "av": ("drone", "autonomous", "flight", "aviation", "robot"),
    "perception": ("vision", "lidar", "perception", "sensor"),
    "industrial": ("vision", "quality", "inspection", "industrial"),
    "neuroscience": ("neuro", "brain", "cognitive"),
}

_SHAPE = (
    "research",
    "perception",
    "robot",
    "machine learning",
    "autonomous",
    "neuro",
    "vision",
    "slam",
    "control",
    "embedded",
    "scientist",
    "intern",
    "working student",
    "werkstudent",
    "thesis",
    "junior",
    "engineer",
)

_GEO = (
    "munich",
    "münchen",
    "muenchen",
    "gilching",
    "garching",
    "germany",
    "deutschland",
    "remote",
    "hybrid",
    "bavaria",
    "bayern",
)

_SKIP_TITLE = re.compile(
    r"\b(sales|account executive|business development|it support|helpdesk|help desk|service desk)\b",
    re.I,
)
_SENIOR = re.compile(r"\b(senior|staff|principal|director|head of|expert)\b", re.I)
_STUDENT = re.compile(r"working student|werkstudent|intern(?:ship)?|\bthesis\b", re.I)
_WORKING_STUDENT = re.compile(r"working student|werkstudent", re.I)
_JUNIOR_WORD = re.compile(r"\bjunior\b", re.I)
_PHD = re.compile(r"\bph\.?d\.?\b", re.I)
_DOMAIN_WORDS = (
    "robot",
    "manipulation",
    "perception",
    "imitation",
    "teleoperation",
    "data",
    "vision",
    "learning",
    "signal",
)


def match_jobs(store: VaultStore) -> str:
    top, fresh, open_count = _ranked(store)
    digest = _render_digest(top, fresh)
    store.write(paths.DIGEST, digest)
    lines = ["match: " + str(len(top))]
    if not top:
        lines.append("no open role passed the hard filters")
    for index, row in enumerate(top, start=1):
        lines.append(
            str(index)
            + ". "
            + str(row["legal_name"])
            + "  "
            + str(row["score"])
            + "  "
            + str(row["why"])
        )
        role = str(row.get("role") or "")
        lines.append("   " + (role if role else "no open role yet"))
    lines.append("new: " + str(len(fresh)))
    for item in fresh:
        lines.append(
            "  "
            + item["legal_name"]
            + " — "
            + item["title"]
            + " — "
            + item["first_seen"]
        )
    if open_count < 10:
        lines.append("open roles: " + str(open_count) + ". /jobs discover")
    return "\n".join(lines)


def queue_top(store: VaultStore) -> tuple[list[dict[str, object]], int]:
    top, _fresh, open_count = _ranked(store)
    return top, open_count


def _ranked(store: VaultStore) -> tuple[list[dict[str, object]], list[dict[str, str]], int]:
    criterion = load_criterion(store)
    weights = criterion["weights"]
    if not isinstance(weights, dict):
        weights = {}
    graduation = str(criterion.get("graduation_date") or "")
    track = str(criterion.get("track") or "student")
    profile = ""
    profile_path = read_profile_path(store)
    if store.exists(profile_path):
        profile = redact_profile(store.read(profile_path)).lower()
    previous = _previous_updated(store)
    rows: list[dict[str, object]] = []
    fresh: list[dict[str, str]] = []
    for rel in store.list_md(paths.COMPANIES):
        text = store.read(rel)
        slug = rel.rsplit("/", 1)[-1].removesuffix(".md")
        company = load_company(text, slug)
        scored = _score_company(company, profile, weights, graduation, track)
        if scored is not None:
            rows.append(scored)
        legal = str(company.get("legal_name") or slug)
        for role in read_roles(str(company.get("body") or "")):
            seen = str(role.get("first_seen") or "")
            if previous and seen > previous and (role.get("status") or "open") == "open":
                fresh.append(
                    {
                        "legal_name": legal,
                        "title": str(role.get("title") or ""),
                        "url": str(role.get("url") or ""),
                        "first_seen": seen,
                    }
                )
    rows.sort(key=_sort_key)
    fresh.sort(key=lambda item: (item["first_seen"], item["legal_name"], item["title"]), reverse=True)
    role_rows = [row for row in rows if str(row.get("role") or "")]
    fillers = [row for row in rows if not str(row.get("role") or "") and _discovered(row)]
    return (role_rows + fillers)[:10], fresh, len(role_rows)


def _score_company(
    company: dict[str, object],
    profile: str,
    weights: dict[str, object],
    graduation: str,
    track: str = "student",
) -> dict[str, object] | None:
    cluster = str(company.get("cluster") or "unclassified")
    munich = str(company.get("munich") or "")
    source = str(company.get("source") or "")
    fit = str(company.get("fit") or "").strip()
    roles = read_roles(str(company.get("body") or ""))
    passed = [role for role in roles if _role_ok(role, munich, graduation, track)]
    discovered = _source_discovered(source)
    if roles and not passed and not discovered:
        return None
    if cluster == "unclassified" and fit == "" and not passed and not discovered:
        return None
    if not passed and munich not in ("yes", "office") and not discovered:
        return None
    cluster_points = _cluster_points(cluster, source, weights)
    preference = 0
    slug = str(company.get("slug") or "")
    if slug not in _BIG and cluster != "unclassified":
        preference = _weight(weights, "preference_small")
    base = [
        ("cluster", cluster_points),
        ("preference", preference),
    ]
    if not passed:
        score = cluster_points + preference
        return {
            "slug": slug,
            "legal_name": str(company.get("legal_name") or slug),
            "score": score,
            "why": _why(base),
            "role": "",
            "url": "",
            "domain_hits": 0,
            "working_student": False,
            "cluster": cluster,
            "source": source,
        }
    best: dict[str, object] | None = None
    for role in passed:
        scored = _score_role(role, profile, cluster, weights, graduation, track, base)
        if best is None or _sort_key(scored) < _sort_key(best):
            best = scored
    if best is None:
        return None
    best["slug"] = slug
    best["legal_name"] = str(company.get("legal_name") or slug)
    best["cluster"] = cluster
    best["source"] = source
    return best


def _role_ok(role: dict[str, str], munich: str, graduation: str, track: str = "student") -> bool:
    title = role.get("title") or ""
    if not title:
        return False
    if (role.get("status") or "open") in ("applied", "skip"):
        return False
    if _SKIP_TITLE.search(title):
        return False
    if _PHD.search(title) and not re.search(r"equivalent", title, re.I):
        return False
    allow_senior = track == "full" or _graduated(graduation)
    if not allow_senior and _SENIOR.search(title) and not _STUDENT.search(title) and not _JUNIOR_WORD.search(title):
        return False
    low = title.lower()
    if not any(needle in low for needle in _SHAPE) and not re.search(r"\bml\b", low):
        return False
    return _geo_text(role.get("location") or "", munich)


def _geo_text(location: str, munich: str) -> bool:
    if not location.strip():
        return munich in ("yes", "office")
    low = location.lower()
    return any(word in low for word in _GEO)


def _graduated(graduation: str) -> bool:
    text = graduation.strip()
    if len(text) < 10:
        return False
    try:
        day = date.fromisoformat(text[:10])
    except ValueError:
        return False
    return date.today() >= day


def _score_role(
    role: dict[str, str],
    profile: str,
    cluster: str,
    weights: dict[str, object],
    graduation: str,
    track: str,
    base: list[tuple[str, int]],
) -> dict[str, object]:
    title = role.get("title") or ""
    hits = _domain_hits(title, profile, cluster)
    domain_points = _weight(weights, "domain") if hits else 0
    seniority_name, seniority_points = _seniority(title, weights, graduation, track)
    parts = list(base)
    if seniority_points:
        parts.append((seniority_name, seniority_points))
    if domain_points:
        parts.append(("domain", domain_points))
    parts.append(("open role", _weight(weights, "open_role")))
    return {
        "score": sum(points for _name, points in parts),
        "why": _why(parts),
        "role": title,
        "url": role.get("url") or "",
        "domain_hits": hits,
        "working_student": bool(_WORKING_STUDENT.search(title)),
    }


def _domain_hits(title: str, profile: str, cluster: str) -> int:
    low = title.lower()
    needles = _CLUSTER_NEEDLES.get(cluster, ())
    profile_fits = any(needle in profile for needle in needles)
    hits = 0
    for word in _DOMAIN_WORDS:
        if word not in low:
            continue
        if profile_fits or word in profile:
            hits += 1
    return hits


def _seniority(title: str, weights: dict[str, object], graduation: str, track: str = "student") -> tuple[str, int]:
    if track == "full":
        if _JUNIOR_WORD.search(title):
            return "junior", _weight(weights, "seniority_junior")
        return "", 0
    if _graduated(graduation):
        return "", 0
    if _STUDENT.search(title):
        return "student", _weight(weights, "seniority_student")
    if _JUNIOR_WORD.search(title):
        return "junior", _weight(weights, "seniority_junior")
    return "", 0


def _sort_key(item: dict[str, object]) -> tuple[int, int, int, str]:
    return (
        -int(item.get("score") or 0),
        -int(item.get("domain_hits") or 0),
        0 if item.get("working_student") else 1,
        str(item.get("slug") or ""),
    )


def _previous_updated(store: VaultStore) -> str:
    if not store.exists(paths.DIGEST):
        return ""
    for raw in store.read(paths.DIGEST).splitlines():
        line = raw.strip()
        if line.startswith("Updated:"):
            return line.split(":", 1)[1].strip()
    return ""


def _source_discovered(source: str) -> bool:
    text = source.lower()
    return "seedtable" in text or text == "discover" or text == "both"


def _discovered(row: dict[str, object]) -> bool:
    return _source_discovered(str(row.get("source") or ""))


def _cluster_points(cluster: str, source: str, weights: dict[str, object]) -> int:
    if cluster != "unclassified":
        return _weight(weights, "cluster_classified")
    if "seedtable" in source or source == "discover":
        return _weight(weights, "cluster_discover")
    return _weight(weights, "cluster_unclassified")
    if cluster != "unclassified":
        return _weight(weights, "cluster_classified")
    if "seedtable" in source or source == "discover":
        return _weight(weights, "cluster_discover")
    return _weight(weights, "cluster_unclassified")


def _weight(weights: dict[str, object], key: str) -> int:
    try:
        return int(weights.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _why(parts: list[tuple[str, int]]) -> str:
    kept = [name + " " + str(points) for name, points in parts if points > 0]
    if not kept:
        return "hard filters only"
    return ", ".join(kept)


def _render_digest(rows: list[dict[str, object]], fresh: list[dict[str, str]]) -> str:
    lines = ["# Digest", "", "Updated: " + date.today().isoformat(), "", "## Top", ""]
    if not rows:
        lines.append("No company passed the hard filters.")
    for index, row in enumerate(rows, start=1):
        lines.append(
            str(index)
            + ". "
            + str(row["legal_name"])
            + " — "
            + str(row["score"])
            + " — "
            + str(row["why"])
        )
        role = str(row.get("role") or "")
        if role:
            lines.append("   - role: " + role)
        url = str(row.get("url") or "")
        if url:
            lines.append("   - url: " + url)
        lines.append("   - slug: " + str(row["slug"]))
    lines.extend(["", "## New", ""])
    if not fresh:
        lines.append("No new postings.")
    for item in fresh:
        lines.append("- company: " + item["legal_name"])
        lines.append("  title: " + item["title"])
        lines.append("  url: " + item["url"])
        lines.append("  first_seen: " + item["first_seen"])
    lines.append("")
    return "\n".join(lines)
