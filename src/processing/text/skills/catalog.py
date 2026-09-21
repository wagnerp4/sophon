from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .spec import SkillSpec, parse_skill

DEFAULT_CATALOG_CHARS = 1500
_SKIP_ROOT_NAMES = frozenset({"skills-cursor"})


@dataclass
class SkillRoot:
    path: Path
    label: str
    writable: bool


@dataclass
class SkillEntry:
    name: str
    description: str
    root_label: str
    skill_dir: Path
    skill_md: Path
    spec: SkillSpec
    writable: bool = False


def _home() -> Path:
    return Path.home()


def skill_write_roots(project_root: Path) -> tuple[Path, Path]:
    return (
        project_root / ".sophon" / "skills",
        _home() / ".sophon" / "skills",
    )


def resolve_roots(project_root: Path) -> list[SkillRoot]:
    roots: list[SkillRoot] = []
    seen: set[str] = set()

    def add(path: Path, label: str, writable: bool) -> None:
        try:
            resolved = path.expanduser()
        except Exception:
            return
        if resolved.name in _SKIP_ROOT_NAMES:
            return
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        roots.append(SkillRoot(path=resolved, label=label, writable=writable))

    override = os.environ.get("SOPHON_SKILLS_DIRS", "").strip()
    if override:
        for i, part in enumerate(override.split(os.pathsep)):
            trimmed = part.strip()
            if trimmed:
                add(Path(trimmed), "override" if i == 0 else f"override{i}", i == 0)
        return roots

    home = _home()
    add(project_root / ".sophon" / "skills", "project", True)
    add(home / ".sophon" / "skills", "user", True)
    add(project_root / ".cursor" / "skills", "cursor:project", False)
    add(home / ".cursor" / "skills", "cursor:user", False)
    add(project_root / ".claude" / "skills", "claude:project", False)
    add(home / ".claude" / "skills", "claude:user", False)
    return roots


def _scan_root(root: SkillRoot) -> list[SkillEntry]:
    entries: list[SkillEntry] = []
    base = root.path
    if not base.is_dir():
        return entries
    try:
        children = sorted(base.iterdir())
    except OSError:
        return entries
    for child in children:
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            text = skill_md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        spec, err = parse_skill(text)
        if spec is None or err is not None:
            continue
        entries.append(
            SkillEntry(
                name=spec.name,
                description=spec.description,
                root_label=root.label,
                skill_dir=child,
                skill_md=skill_md,
                spec=spec,
                writable=root.writable,
            )
        )
    return entries


def catalog_chars_from_env() -> int:
    raw = os.environ.get("SOPHON_SKILLS_CATALOG_CHARS", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_CATALOG_CHARS
    except ValueError:
        value = DEFAULT_CATALOG_CHARS
    return max(200, value)


@dataclass
class SkillCatalog:
    project_root: Path
    entries: list[SkillEntry] = field(default_factory=list)
    shadowed: list[tuple[str, str, str]] = field(default_factory=list)
    _by_name: dict[str, SkillEntry] = field(default_factory=dict)

    def refresh(self) -> "SkillCatalog":
        by_name: dict[str, SkillEntry] = {}
        shadowed: list[tuple[str, str, str]] = []
        for root in resolve_roots(self.project_root):
            for entry in _scan_root(root):
                existing = by_name.get(entry.name)
                if existing is not None:
                    shadowed.append((entry.name, entry.root_label, existing.root_label))
                    continue
                by_name[entry.name] = entry
        self.entries = sorted(by_name.values(), key=lambda e: e.name)
        self.shadowed = shadowed
        self._by_name = by_name
        return self

    def get(self, name: str) -> SkillEntry | None:
        return self._by_name.get(str(name or "").strip().lower())

    def names(self) -> list[str]:
        return [entry.name for entry in self.entries]

    def catalog_lines(self, cap_chars: int | None = None) -> tuple[list[str], int]:
        cap = catalog_chars_from_env() if cap_chars is None else cap_chars
        lines: list[str] = []
        used = 0
        dropped = 0
        for entry in self.entries:
            desc = " ".join(entry.description.split())
            line = f"- {entry.name} (root={entry.root_label}): {desc}"
            cost = len(line) + 1
            if used + cost > cap and lines:
                dropped += 1
                continue
            lines.append(line)
            used += cost
        return lines, dropped
