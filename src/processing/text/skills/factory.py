from __future__ import annotations

import os
from pathlib import Path

from .catalog import SkillCatalog

_DISABLE = frozenset({"0", "off", "false", "no", "disabled"})


def skills_enabled() -> bool:
    raw = os.environ.get("SOPHON_SKILLS", "1").strip().lower()
    return raw not in _DISABLE


def open_skill_catalog(project_root: Path | None = None) -> SkillCatalog | None:
    if not skills_enabled():
        return None
    if project_root is None:
        from utils.device.env_bootstrap import sophon_project_root

        project_root = sophon_project_root()
    catalog = SkillCatalog(project_root=Path(project_root))
    catalog.refresh()
    return catalog
