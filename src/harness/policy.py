from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

HARNESS_DIRNAME = ".sophon"
HARNESS_FILE = "harness.yaml"
HARNESS_LOCAL_FILE = "harness.local.yaml"
MODES = ("plan", "chat", "agent")

@dataclass
class Policy:
    workspace: str = "."
    mode: str = "agent"
    sandbox: str = "full"
    allow: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)

    def workspace_path(self, project_root: Path) -> Path:
        raw = (self.workspace or ".").strip() or "."
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = Path(project_root) / target
        try:
            return target.resolve()
        except OSError:
            return target


def default_policy() -> Policy:
    return Policy(
        workspace=".",
        mode="agent",
        allow=[],
        ask=[],
        deny=[],
    )


def harness_dir(project_root: Path) -> Path:
    return Path(project_root) / HARNESS_DIRNAME


def harness_yaml_path(project_root: Path) -> Path:
    return harness_dir(project_root) / HARNESS_FILE


def harness_local_path(project_root: Path) -> Path:
    return harness_dir(project_root) / HARNESS_LOCAL_FILE


def policy_file_paths(project_root: Path) -> tuple[Path, Path]:
    return harness_yaml_path(project_root), harness_local_path(project_root)


def load_policy(project_root: Path) -> Policy:
    merged = default_policy()
    root = Path(project_root)
    for path in policy_file_paths(root):
        overlay = _read_policy_file(path)
        if overlay is not None:
            merged = merge_policy(merged, overlay, replace_lists=False)
    return merged


def merge_policy(base: Policy, overlay: Policy, *, replace_lists: bool = False) -> Policy:
    if replace_lists:
        allow = list(overlay.allow)
        ask = list(overlay.ask)
        deny = list(overlay.deny)
    else:
        allow = _unique(list(base.allow) + list(overlay.allow))
        ask = _unique(list(base.ask) + list(overlay.ask))
        deny = _unique(list(base.deny) + list(overlay.deny))
    mode = overlay.mode if overlay.mode in MODES else base.mode
    workspace = overlay.workspace if overlay.workspace else base.workspace
    sandbox = overlay.sandbox if overlay.sandbox else base.sandbox
    return Policy(workspace=workspace, mode=mode, sandbox=sandbox, allow=allow, ask=ask, deny=deny)


def policy_from_mapping(data: dict[str, Any] | None) -> Policy:
    payload = data if isinstance(data, dict) else {}
    perms = payload.get("permissions")
    if not isinstance(perms, dict):
        perms = {}
    mode = str(payload.get("mode") or "agent").strip().lower()
    if mode not in MODES:
        mode = "agent"
    from integrations.shell.sandbox import normalize_sandbox

    sandbox = normalize_sandbox(payload.get("sandbox")) if "sandbox" in payload else ""
    workspace = str(payload.get("workspace") or ".").strip() or "."
    return Policy(
        workspace=workspace,
        mode=mode,
        sandbox=sandbox,
        allow=_string_list(perms.get("allow")),
        ask=_string_list(perms.get("ask")),
        deny=_string_list(perms.get("deny")),
    )


def policy_to_mapping(policy: Policy) -> dict[str, Any]:
    return {
        "workspace": policy.workspace,
        "mode": policy.mode,
        "sandbox": policy.sandbox,
        "permissions": {
            "allow": list(policy.allow),
            "ask": list(policy.ask),
            "deny": list(policy.deny),
        },
    }


def append_allow_rules(path: Path, rules: list[str]) -> list[str]:
    added: list[str] = []
    data: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    perms = data.get("permissions")
    if not isinstance(perms, dict):
        perms = {}
        data["permissions"] = perms
    allow = _string_list(perms.get("allow"))
    for rule in rules:
        text = str(rule).strip()
        if not text or text in allow:
            continue
        allow.append(text)
        added.append(text)
    perms["allow"] = allow
    if "workspace" not in data:
        data["workspace"] = "."
    if "mode" not in data:
        data["mode"] = "agent"
    path.parent.mkdir(parents=True, exist_ok=True)
    dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.write_text(dumped, encoding="utf-8")
    return added


def is_policy_path(path: Path, project_root: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser()
    protected = []
    for candidate in policy_file_paths(project_root):
        try:
            protected.append(candidate.resolve())
        except OSError:
            protected.append(candidate)
    return resolved in protected


def _read_policy_file(path: Path) -> Policy | None:
    if not path.is_file():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(loaded, dict):
        return None
    return policy_from_mapping(loaded)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
