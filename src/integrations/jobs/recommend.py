from __future__ import annotations

import re

from integrations.jobs import paths
from integrations.jobs.feeds import read_roles
from integrations.jobs.ingest import read_profile_path
from integrations.jobs.notes import load_company, slugify
from integrations.jobs.profile import redact_profile
from integrations.jobs.store import VaultStore

_TERMS = (
    "ros",
    "slam",
    "pytorch",
    "lidar",
    "computer vision",
    "reinforcement learning",
    "embedded",
    "cuda",
    "transformer",
    "sensor fusion",
    "kinematics",
    "eeg",
    "drone",
    "perception",
    "manipulation",
    "planning",
)


def recommend_plan(store: VaultStore, token: str, state: object | None = None) -> str:
    company, role = _select(store, token)
    if company is None:
        return "recommend: name a digest row or a company slug"
    profile = ""
    profile_path = read_profile_path(store)
    if store.exists(profile_path):
        profile = redact_profile(store.read(profile_path))
    skeleton = _skeleton(company, role, profile)
    body = _with_model(state, skeleton, profile, company, role)
    slug = str(company.get("slug") or "plan")
    rel = paths.PLANS + "/" + slug + ".md"
    store.write(rel, body)
    return "wrote " + rel


def _select(store: VaultStore, token: str) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    text = token.strip()
    if not text:
        text = "1"
    if text.isdigit() and store.exists(paths.DIGEST):
        slug = _digest_slug(store.read(paths.DIGEST), int(text))
        if slug:
            text = slug
    slug = slugify(text.split()[0])
    rel = paths.COMPANIES + "/" + slug + ".md"
    if not store.exists(rel):
        found = _find_slug(store, slug)
        if found is None:
            return None, None
        rel = found
        slug = rel.rsplit("/", 1)[-1].removesuffix(".md")
    company = load_company(store.read(rel), slug)
    rest = " ".join(text.split()[1:]).lower()
    roles = read_roles(str(company.get("body") or ""))
    chosen = None
    if rest:
        for role in roles:
            if rest in (role.get("title") or "").lower():
                chosen = role
                break
    elif roles:
        chosen = roles[0]
    return company, chosen


def _digest_slug(text: str, index: int) -> str:
    current = 0
    slug = ""
    for raw in text.splitlines():
        line = raw.strip()
        if re.match(r"^\d+\. ", line):
            current = int(line.split(".", 1)[0])
            slug = ""
        if line.startswith("- slug:") or line.startswith("slug:"):
            slug = line.split(":", 1)[1].strip()
            if current == index:
                return slug
    return ""


def _find_slug(store: VaultStore, needle: str) -> str | None:
    for rel in store.list_md(paths.COMPANIES):
        slug = rel.rsplit("/", 1)[-1].removesuffix(".md")
        if slug == needle:
            return rel
        company = load_company(store.read(rel), slug)
        if slugify(str(company.get("legal_name") or "")) == needle:
            return rel
    return None


def _skeleton(
    company: dict[str, object],
    role: dict[str, str] | None,
    profile: str,
) -> str:
    legal = str(company.get("legal_name") or company.get("slug") or "")
    title = ""
    blob = legal
    if role:
        title = role.get("title") or ""
        blob = " ".join(
            [
                title,
                role.get("location") or "",
                role.get("team") or "",
            ]
        )
    low_profile = profile.lower()
    # TODO: feeds store title, location, and team. Tool gaps stay thin until the posting body is kept.
    gaps = [term for term in _TERMS if term in blob.lower() and term not in low_profile]
    lines = ["# Learning plan: " + legal, ""]
    if title:
        lines.append("Role: " + title)
        lines.append("")
    lines.append("## Gaps")
    lines.append("")
    if gaps:
        for term in gaps:
            lines.append("- " + term + " is in the posting and not in the profile excerpt.")
    else:
        lines.append("- No listed tool from the posting is missing in the profile excerpt.")
        lines.append("- The gap is evidence: map ICS work and the three publications onto this role in one page.")
    lines.extend(["", "## Study", ""])
    if gaps:
        for index, term in enumerate(gaps, start=1):
            lines.append(
                str(index)
                + ". Build or revise a small demo for "
                + term
                + " that cites the existing ICS work."
            )
    else:
        lines.append("1. Rewrite the publication list as a one-page map onto this role.")
        lines.append("2. Put the two GitHub accounts next to the one system this team ships.")
    lines.extend(
        [
            "",
            "## Skip",
            "",
            "- Roles that require a finished PhD, unless the posting says equivalent research.",
            "- Senior, staff, or director titles while graduation_date in Criterion.md is empty.",
            "- Sales and IT support tracks at this company.",
            "",
            "This note is a study plan. It does not submit an application.",
            "",
        ]
    )
    return "\n".join(lines)


def _with_model(
    state: object | None,
    skeleton: str,
    profile: str,
    company: dict[str, object],
    role: dict[str, str] | None,
) -> str:
    if state is None:
        return skeleton
    backend = str(getattr(state, "backend_id", "") or "")
    if backend in ("", "hf") and getattr(state, "model", None) is None:
        return skeleton + "\nModel weights are not loaded. The sections above are the plan.\n"
    # TODO: stream this completion into the chat transcript instead of blocking the slash command
    prompt = (
        "Write a short recruitment study plan. Use only the profile excerpt and the posting. "
        "Do not invent contact details. Keep the headings Gaps, Study, and Skip.\n\n"
        "Company: "
        + str(company.get("legal_name") or "")
        + "\nRole: "
        + str((role or {}).get("title") or "")
        + "\n\nProfile excerpt:\n"
        + profile[:2500]
        + "\n\nDraft:\n"
        + skeleton
    )
    params = getattr(state, "params", None)
    old_max = getattr(params, "max_new_tokens", None) if params is not None else None
    old_purpose = getattr(state, "_energy_purpose", None)
    try:
        if params is not None and old_max is not None:
            params.max_new_tokens = min(int(old_max), 700)
        setattr(state, "_energy_purpose", "jobs-recommend")
        from cli.agent_runtime import complete_chat_turn

        result = complete_chat_turn(
            state,
            [
                {
                    "role": "system",
                    "content": "You write a study plan for one applicant. You do not submit applications.",
                },
                {"role": "user", "content": prompt},
            ],
            None,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return skeleton + "\nModel call failed (" + str(exc) + "). The sections above are the plan.\n"
    finally:
        if params is not None and old_max is not None:
            params.max_new_tokens = old_max
        if old_purpose is None:
            if hasattr(state, "_energy_purpose"):
                try:
                    delattr(state, "_energy_purpose")
                except AttributeError:
                    pass
        else:
            setattr(state, "_energy_purpose", old_purpose)
    text = str(getattr(result, "text", "") or "").strip()
    if not text or text.startswith("(energy stop"):
        extra = text or "model returned an empty plan"
        return skeleton + "\n" + extra + "\n"
    if "Gaps" not in text or "Study" not in text:
        return skeleton + "\n\n## Model note\n\n" + text + "\n"
    legal = str(company.get("legal_name") or "")
    return "# Learning plan: " + legal + "\n\n" + text.rstrip() + "\n"
