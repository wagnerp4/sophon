from __future__ import annotations

import shlex

from integrations.jobs import paths
from integrations.jobs.discover import discover_add, discover_jobs, discover_skip
from integrations.jobs.feeds import fetch_roles, resolve_feeds
from integrations.jobs.ingest import ingest_lists
from integrations.jobs.match import match_jobs
from integrations.jobs.recommend import recommend_plan
from integrations.jobs.store import VaultStore

_HELP = """/jobs
/jobs track student|full
/jobs ingest [PROFILE LIST...]
/jobs discover
/jobs discover add N|SLUG
/jobs discover skip N|SLUG
/jobs match
/jobs recommend [N|SLUG]
/jobs fetch

/jobs opens the top-10 queue. Enter or click a row to mark applied or skip.
track student|full switches seniority without editing weights.
/jobs discover add N adds the first N pending names. /jobs discover add SLUG adds one name.
When fewer than 10 open roles remain, discover is the refill.
match scores open roles and writes Digest.md.
fetch reads public ATS feeds for companies already on the list, then matches again."""


def handle_jobs_command(arg: str, state: object | None = None) -> str:
    parts = _split(arg)
    if not parts:
        text = match_jobs(VaultStore())
        return "queue\n" + text
    head = parts[0].lower()
    store = VaultStore()
    if head == "track":
        from integrations.jobs.criterion import set_track

        track = parts[1].lower() if len(parts) > 1 else ""
        if track not in ("student", "full"):
            return "usage: /jobs track student|full"
        set_track(store, track)
        return "track: " + track + "\n" + match_jobs(store)
    if head == "ingest":
        profile, lists = _ingest_args(parts[1:])
        return ingest_lists(store, profile_path=profile, list_paths=lists)
    if head == "discover":
        if len(parts) == 1:
            return discover_jobs(store)
        action = parts[1].lower()
        token = " ".join(parts[2:]) if len(parts) > 2 else ""
        if action == "add":
            if not token:
                return "usage: /jobs discover add N|SLUG"
            return discover_add(store, token)
        if action in ("skip", "no"):
            if not token:
                return "usage: /jobs discover skip N|SLUG"
            return discover_skip(store, token)
        return "usage: /jobs discover [add|skip N|SLUG]"
    if head == "match":
        return match_jobs(store)
    if head == "recommend":
        token = " ".join(parts[1:]) if len(parts) > 1 else "1"
        return recommend_plan(store, token, state)
    if head == "fetch":
        resolved = resolve_feeds(store)
        fetched = fetch_roles(store)
        matched = match_jobs(store)
        return resolved + "\n" + fetched + "\n" + matched
    if head in ("help", "-h"):
        return _HELP
    return "unknown /jobs mode " + head + "\n" + _HELP


def _ingest_args(parts: list[str]) -> tuple[str, list[str] | None]:
    if not parts:
        return paths.PROFILE, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[1:]


def _split(arg: str) -> list[str]:
    text = arg.strip()
    if not text:
        return []
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return text.split()
