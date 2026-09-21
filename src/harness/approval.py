from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DECISION_ONCE = "once"
DECISION_PERSIST = "persist"
DECISION_DENY = "deny"

VERDICT_ALLOW = "allow"
VERDICT_DENY = "deny"
VERDICT_ASK = "ask"

DENIED_PREFIX = "error: permission denied"


@dataclass
class PermissionRequest:
    tool: str
    summary: str
    persist_rules: list[str]
    cwd: str = ""
    workspace: str = ""
    reason: str = ""


@dataclass
class AuthResult:
    allowed: bool
    verdict: str
    reason: str
    permission: str
    persist_rules: list[str] = field(default_factory=list)
    request: PermissionRequest | None = None
    message: str = ""

    @classmethod
    def allow(cls, reason: str = "allow") -> "AuthResult":
        return cls(allowed=True, verdict=VERDICT_ALLOW, reason=reason, permission="allow")

    @classmethod
    def deny(cls, reason: str) -> "AuthResult":
        return cls(
            allowed=False,
            verdict=VERDICT_DENY,
            reason=reason,
            permission="deny",
            message=f"{DENIED_PREFIX} ({reason})",
        )

    @classmethod
    def ask(
        cls,
        request: PermissionRequest,
        *,
        reason: str,
    ) -> "AuthResult":
        return cls(
            allowed=False,
            verdict=VERDICT_ASK,
            reason=reason,
            permission="ask",
            persist_rules=list(request.persist_rules),
            request=request,
            message=f"{DENIED_PREFIX} ({reason})",
        )


def path_in_workspace(path: Path | None, workspace: Path) -> bool:
    if path is None:
        return True
    try:
        resolved = path.expanduser().resolve()
        root = workspace.expanduser().resolve()
    except OSError:
        resolved = path.expanduser()
        root = workspace
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False
