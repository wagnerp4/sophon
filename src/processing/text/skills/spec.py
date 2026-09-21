from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

MAX_NAME = 64
MAX_DESCRIPTION = 1024
MAX_COMPATIBILITY = 500

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"^\ufeff?---[ \t]*\n(.*?)\n---[ \t]*(?:\n(.*))?$", re.DOTALL)


@dataclass
class SkillSpec:
    name: str
    description: str
    body: str = ""
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None
    disable_model_invocation: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def split_frontmatter(text: str) -> tuple[dict[str, Any], str] | None:
    match = _FRONTMATTER_RE.match(text or "")
    if match is None:
        return None
    raw = match.group(1) or ""
    body = match.group(2) or ""
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    return data, body


def validate_name(name: str) -> str | None:
    if not name:
        return "name is required"
    if len(name) > MAX_NAME:
        return f"name exceeds {MAX_NAME} characters"
    if "--" in name:
        return "name must not contain consecutive hyphens"
    if not _NAME_RE.match(name):
        return "name must be lowercase letters, numbers, and single hyphens"
    return None


def validate_description(description: str) -> str | None:
    if not description or not description.strip():
        return "description is required"
    if len(description) > MAX_DESCRIPTION:
        return f"description exceeds {MAX_DESCRIPTION} characters"
    return None


def _coerce_metadata(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        out[str(key)] = str(value)
    return out


def _truthy(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


_KNOWN_KEYS = frozenset(
    {
        "name",
        "description",
        "license",
        "compatibility",
        "metadata",
        "allowed-tools",
        "disable-model-invocation",
    }
)


def parse_skill(text: str) -> tuple[SkillSpec | None, str | None]:
    split = split_frontmatter(text)
    if split is None:
        return None, "missing YAML frontmatter"
    data, body = split
    name = str(data.get("name") or "").strip()
    description = str(data.get("description") or "").strip()
    err = validate_name(name) or validate_description(description)
    if err is not None:
        return None, err
    compatibility = data.get("compatibility")
    compatibility_s = str(compatibility).strip() if compatibility is not None else None
    if compatibility_s and len(compatibility_s) > MAX_COMPATIBILITY:
        return None, f"compatibility exceeds {MAX_COMPATIBILITY} characters"
    allowed = data.get("allowed-tools")
    allowed_s = str(allowed).strip() if allowed is not None else None
    license_raw = data.get("license")
    license_s = str(license_raw).strip() if license_raw is not None else None
    extra = {str(k): v for k, v in data.items() if str(k) not in _KNOWN_KEYS}
    spec = SkillSpec(
        name=name,
        description=description,
        body=body.strip("\n"),
        license=license_s,
        compatibility=compatibility_s,
        metadata=_coerce_metadata(data.get("metadata")),
        allowed_tools=allowed_s,
        disable_model_invocation=_truthy(data.get("disable-model-invocation")),
        extra=extra,
    )
    return spec, None


def render_skill_md(
    name: str,
    description: str,
    body: str,
    *,
    license: str | None = None,
    compatibility: str | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    lines = ["---", f"name: {name}", f"description: {description}"]
    if license:
        lines.append(f"license: {license}")
    if compatibility:
        lines.append(f"compatibility: {compatibility}")
    if metadata:
        lines.append("metadata:")
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")
    lines.append("---")
    front = "\n".join(lines)
    body_clean = body.strip("\n")
    return f"{front}\n\n{body_clean}\n"
