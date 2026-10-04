from __future__ import annotations

import re

from integrations.jobs import paths
from integrations.jobs.notes import (
    CompanySeed,
    company_path,
    parse_classified,
    parse_unclassified,
    render_company,
    slugify,
)
from integrations.jobs.store import VaultStore

_DEFAULT_CRITERION = """# Criterion

## Hard filters

- Geography: Munich metro, or Germany-remote with a Munich office. Gilching counts.
- Role shape: research, perception, robotics software, ML, AV, neurotech. Working-student, thesis, internship, and junior research before a full-time senior title.
- Skip: pure sales, pure IT support, roles that require a finished PhD unless the posting says equivalent research.

## Graduation

graduation_date:

## Profile

profile_path: \"Personal KB/About.md\"

## Check

check: daily
track: student

## Weights

- evidence: 30
- cluster_classified: 40
- cluster_unclassified: 15
- cluster_discover: 5
- preference_small: 20
- open_role: 10
- seniority_student: 40
- seniority_junior: 15
- domain: 30
"""


def ingest_lists(
    store: VaultStore,
    *,
    profile_path: str = paths.PROFILE,
    list_paths: list[str] | None = None,
) -> str:
    lists = list_paths if list_paths else [paths.CLASSIFIED, paths.UNCLASSIFIED]
    seeds: list[CompanySeed] = []
    used: list[str] = []
    for rel in lists:
        text = store.read(rel)
        used.append(rel)
        if "unclassified" in rel.lower():
            seeds.extend(parse_unclassified(text))
        else:
            seeds.extend(parse_classified(text))
    merged = _merge_seeds(seeds)
    written = 0
    for seed in merged:
        rel = company_path(paths.COMPANIES, seed.slug)
        existing = None
        if store.exists(rel):
            existing = store.read(rel)
        store.write(rel, render_company(seed, existing))
        written += 1
    _write_criterion(store, profile_path)
    _write_seedtable_stub(store)
    return (
        "ingested "
        + str(written)
        + " companies from "
        + ", ".join(used)
        + "\nprofile: "
        + profile_path
    )


def _merge_seeds(seeds: list[CompanySeed]) -> list[CompanySeed]:
    by_slug: dict[str, CompanySeed] = {}
    for seed in seeds:
        slug = slugify(seed.legal_name) if seed.slug == slugify(seed.legal_name) else seed.slug
        seed.slug = slug
        current = by_slug.get(seed.slug)
        if current is None:
            by_slug[seed.slug] = seed
            continue
        if seed.cluster != "unclassified":
            current.cluster = seed.cluster
            current.munich = "yes"
            if seed.legal_name:
                current.legal_name = seed.legal_name
        if seed.note and seed.note not in current.note:
            current.note = (current.note + " " + seed.note).strip()
        for alias in seed.aliases + [seed.legal_name]:
            if alias and alias != current.legal_name and alias not in current.aliases:
                current.aliases.append(alias)
        if seed.munich == "yes":
            current.munich = "yes"
    return list(by_slug.values())


def _write_criterion(store: VaultStore, profile_path: str) -> None:
    if store.exists(paths.CRITERION):
        text = store.read(paths.CRITERION)
        if "profile_path:" in text:
            text = re.sub(
                r"profile_path:.*",
                "profile_path: " + _quote(profile_path),
                text,
                count=1,
            )
        else:
            text = text.rstrip() + "\n\n## Profile\n\nprofile_path: " + _quote(profile_path) + "\n"
        store.write(paths.CRITERION, text)
        return
    body = _DEFAULT_CRITERION.replace(
        "profile_path: \"Personal KB/About.md\"",
        "profile_path: " + _quote(profile_path),
    )
    store.write(paths.CRITERION, body)


def _write_seedtable_stub(store: VaultStore) -> None:
    from datetime import date

    name = "munich-ai-" + date.today().isoformat() + ".md"
    rel = paths.SEEDTABLE + "/" + name
    existing = store.list_md(paths.SEEDTABLE)
    if any(path.split("/")[-1].startswith("munich-ai-") for path in existing):
        return
    content = (
        "# Munich AI\n\n"
        "source: https://seedtable.com/best-ai-startups-in-munich\n"
        "status: paste names under Names. A direct fetch of the page is blocked.\n\n"
        "## Names\n"
    )
    store.write(rel, content)


def _quote(value: str) -> str:
    return "\"" + value.replace("\"", "") + "\""


def read_profile_path(store: VaultStore) -> str:
    if not store.exists(paths.CRITERION):
        return paths.PROFILE
    text = store.read(paths.CRITERION)
    for line in text.splitlines():
        if line.strip().startswith("profile_path:"):
            value = line.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value
    return paths.PROFILE
