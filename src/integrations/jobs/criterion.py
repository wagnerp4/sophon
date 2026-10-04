from __future__ import annotations

import re

from integrations.jobs import paths
from integrations.jobs.store import VaultStore

_DEFAULT_WEIGHTS = {
    "evidence": 30,
    "cluster_classified": 40,
    "cluster_unclassified": 15,
    "cluster_discover": 5,
    "preference_small": 20,
    "open_role": 10,
    "seniority_student": 40,
    "seniority_junior": 15,
    "domain": 30,
}


def load_criterion(store: VaultStore) -> dict[str, object]:
    text = ""
    if store.exists(paths.CRITERION):
        text = store.read(paths.CRITERION)
    weights = dict(_DEFAULT_WEIGHTS)
    for key in list(weights):
        match = re.search(r"^-\s+" + re.escape(key) + r":\s+(\d+)\s*$", text, re.M)
        if match:
            weights[key] = int(match.group(1))
    graduation = ""
    check = "daily"
    track = "student"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("graduation_date:"):
            graduation = stripped.split(":", 1)[1].strip().strip("\"'")
        elif stripped.startswith("check:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'").lower()
            if value in ("daily", "weekly"):
                check = value
        elif stripped.startswith("track:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'").lower()
            if value in ("student", "full"):
                track = value
    return {
        "weights": weights,
        "graduation_date": graduation,
        "check": check,
        "track": track,
        "text": text,
    }


def set_track(store: VaultStore, track: str) -> str:
    if track not in ("student", "full"):
        raise ValueError("track must be student or full")
    text = store.read(paths.CRITERION) if store.exists(paths.CRITERION) else ""
    line = "track: " + track
    if re.search(r"^track:.*$", text, re.M):
        text = re.sub(r"^track:.*$", line, text, count=1, flags=re.M)
    elif "check:" in text:
        text = re.sub(r"^(check:.*)$", r"\1\n" + line, text, count=1, flags=re.M)
    else:
        text = text.rstrip() + "\n\ntrack: " + track + "\n"
    store.write(paths.CRITERION, text if text.endswith("\n") else text + "\n")
    return track
