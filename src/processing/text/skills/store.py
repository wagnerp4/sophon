from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .catalog import skill_write_roots
from .spec import parse_skill, render_skill_md, validate_description, validate_name

_BODY_TEMPLATE = (
    "# {title}\n\n"
    "## Instructions\n\n"
    "Step-by-step guidance for the agent. Keep it short. Move long reference "
    "material into references/ and load it only when needed.\n\n"
    "## Examples\n\n"
    "Concrete input and output examples.\n"
)


@dataclass
class SkillProposal:
    path: Path
    before: str
    after: str
    name: str


def _title_from_name(name: str) -> str:
    return name.replace("-", " ").strip().title()


def _write_root(project_root: Path, user: bool) -> Path:
    project, user_root = skill_write_roots(project_root)
    return user_root if user else project


def scaffold_skill(
    project_root: Path,
    name: str,
    description: str | None = None,
    *,
    user: bool = False,
) -> SkillProposal:
    clean = str(name or "").strip().lower()
    err = validate_name(clean)
    if err is not None:
        raise ValueError(err)
    desc = str(description or "").strip() or (
        f"{_title_from_name(clean)} skill. Describe what it does and when to use it."
    )
    derr = validate_description(desc)
    if derr is not None:
        raise ValueError(derr)
    dest = _write_root(project_root, user) / clean / "SKILL.md"
    before = dest.read_text(encoding="utf-8", errors="replace") if dest.is_file() else ""
    after = render_skill_md(clean, desc, _BODY_TEMPLATE.format(title=_title_from_name(clean)))
    return SkillProposal(path=dest, before=before, after=after, name=clean)


def import_skill(
    project_root: Path,
    source: str | Path,
    *,
    user: bool = False,
) -> SkillProposal:
    src = Path(str(source)).expanduser()
    skill_md = src / "SKILL.md" if src.is_dir() else src
    if not skill_md.is_file():
        raise ValueError(f"no SKILL.md at {skill_md}")
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    spec, perr = parse_skill(text)
    if spec is None or perr is not None:
        raise ValueError(f"invalid skill: {perr}")
    dest = _write_root(project_root, user) / spec.name / "SKILL.md"
    before = dest.read_text(encoding="utf-8", errors="replace") if dest.is_file() else ""
    return SkillProposal(path=dest, before=before, after=text, name=spec.name)
