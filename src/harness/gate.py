from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.approval import (
    DECISION_DENY,
    DECISION_ONCE,
    DECISION_PERSIST,
    AuthResult,
    PermissionRequest,
    path_in_workspace,
)
from harness.match import (
    allow_command,
    any_command_match,
    infer_persist_rules,
    rule_matches,
    split_compound,
)
from harness.policy import (
    Policy,
    append_allow_rules,
    harness_local_path,
    is_policy_path,
    load_policy,
    policy_file_paths,
)

GATED_TOOLS = frozenset(
    {"shell_exec", "editor_propose_edit", "memory_propose", "subagent", "subagent_fork", "github_create"}
)
PLAN_LIKE_MODES = frozenset({"plan", "chat"})
# TODO: net allow-list is separate from the filesystem sandbox.

_CIRCUIT_PATTERNS = (
    re.compile(r"\brm\b[^;&\n]*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+[/~](?:\s|$)", re.IGNORECASE),
    re.compile(r"\brm\b[^;&\n]*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+[/~](?:\s|$)", re.IGNORECASE),
    re.compile(r"\brm\s+-[a-zA-Z]*rf[a-zA-Z]*\s+/(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bformat-volume\b", re.IGNORECASE),
    re.compile(r"\bclear-disk\b", re.IGNORECASE),
    re.compile(r"\b(stop-computer|restart-computer)\b", re.IGNORECASE),
    re.compile(r"(?:^|\s)shutdown(?:\s|$)", re.IGNORECASE),
    re.compile(
        r"remove-item\b[^;&\n]*-recurse[^;&\n]*(?:\s|^)(?:[a-z]:\\|/|~|\$home)(?:\s|$)",
        re.IGNORECASE,
    ),
)


AskFn = Callable[[PermissionRequest], str]


@dataclass
class Harness:
    project_root: Path
    policy: Policy
    mode_override: str | None = None
    sandbox_override: str | None = None
    session_allow: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if self.mode_override in ("plan", "chat", "agent"):
            return self.mode_override
        return self.policy.mode if self.policy.mode in ("plan", "chat", "agent") else "agent"

    @property
    def sandbox(self) -> str:
        from integrations.shell.sandbox import normalize_sandbox

        if self.sandbox_override:
            return self.sandbox_override
        return normalize_sandbox(getattr(self.policy, "sandbox", "full"))

    def set_sandbox(self, sandbox: str) -> str:
        from integrations.shell.sandbox import SANDBOX_MODES, normalize_sandbox

        token = str(sandbox or "").strip().lower()
        value = normalize_sandbox(token)
        if value != token and token not in ("readonly", "workspace", "qemu"):
            raise ValueError(f"sandbox must be one of: {', '.join(SANDBOX_MODES)}")
        self.sandbox_override = value
        return value

    def set_mode(self, mode: str) -> str:
        token = str(mode or "").strip().lower()
        if token not in ("plan", "chat", "agent"):
            raise ValueError("mode must be plan, chat, or agent")
        self.mode_override = token
        return token

    def workspace_path(self) -> Path:
        return self.policy.workspace_path(self.project_root)

    def reload(self) -> None:
        current_mode = self.mode_override
        self.policy = load_policy(self.project_root)
        self.mode_override = current_mode

    def persist_rules(self, rules: list[str]) -> list[str]:
        added = append_allow_rules(harness_local_path(self.project_root), rules)
        for rule in added:
            if rule not in self.policy.allow:
                self.policy.allow.append(rule)
            if rule not in self.session_allow:
                self.session_allow.append(rule)
        return added

    def describe(self) -> str:
        workspace = self.workspace_path()
        shared, local = policy_file_paths(self.project_root)
        lines = [
            f"mode: {self.mode}",
            f"sandbox: {self.sandbox}",
            f"workspace: {workspace}",
            f"policy: {shared} ({'yes' if shared.is_file() else 'missing'})",
            f"local: {local} ({'yes' if local.is_file() else 'missing'})",
            f"allow: {_format_rules(self.policy.allow)}",
            f"ask: {_format_rules(self.policy.ask)}",
            f"deny: {_format_rules(self.policy.deny)}",
        ]
        if self.session_allow:
            lines.append(f"session allow: {_format_rules(self.session_allow)}")
        return "\n".join(lines)

    def authorize(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        cwd: Path | None = None,
        target: Path | None = None,
        ask: AskFn | None = None,
    ) -> AuthResult:
        args = arguments if isinstance(arguments, dict) else {}
        command = str(args.get("command") or "").strip()
        cwd_path = Path(cwd) if cwd is not None else Path.cwd()
        workspace = self.workspace_path()
        result = self._evaluate(
            tool,
            command=command,
            cwd=cwd_path,
            target=target,
            workspace=workspace,
        )
        if result.verdict != "ask" or result.request is None:
            return result
        decision = ask(result.request) if ask is not None else DECISION_DENY
        if decision == DECISION_PERSIST:
            self.persist_rules(result.persist_rules)
            result.allowed = True
            result.permission = "ask:persist"
            result.message = ""
            return result
        if decision == DECISION_ONCE:
            result.allowed = True
            result.permission = "ask:once"
            result.message = ""
            return result
        result.allowed = False
        result.permission = "ask:deny"
        result.message = "error: permission denied (declined)"
        return result

    def _evaluate(
        self,
        tool: str,
        *,
        command: str,
        cwd: Path,
        target: Path | None,
        workspace: Path,
    ) -> AuthResult:
        persist = infer_persist_rules(tool, command=command, target=target)
        request = PermissionRequest(
            tool=tool,
            summary=_summary(tool, command=command, target=target),
            persist_rules=persist,
            cwd=str(cwd),
            workspace=str(workspace),
            reason="",
        )
        if tool in ("editor_propose_edit", "memory_propose") and target is not None:
            if is_policy_path(target, self.project_root):
                return AuthResult.deny("harness policy files cannot be edited by the model")
        if self.mode in PLAN_LIKE_MODES and _is_gated(tool, target, workspace):
            return AuthResult.deny(f"{self.mode} mode")
        if tool == "shell_exec" and _circuit_breaker(command):
            request.reason = "circuit breaker"
            return AuthResult.ask(request, reason="circuit breaker")
        if _matches_deny(self.policy.deny, tool, command=command, target=target):
            return AuthResult.deny("deny rule")
        if _matches_ask(self.policy.ask, tool, command=command, target=target):
            request.reason = "ask rule"
            return AuthResult.ask(request, reason="ask rule")
        if _matches_allow(self.policy.allow + self.session_allow, tool, command=command, target=target):
            return AuthResult.allow("allow rule")
        if tool == "shell_exec" and self.sandbox == "vm":
            return AuthResult.allow("vm sandbox")
        if tool == "shell_exec":
            request.reason = "shell_exec"
            return AuthResult.ask(request, reason="shell_exec")
        if tool == "github_create":
            request.reason = "github_create"
            return AuthResult.ask(request, reason="github_create")
        if tool in ("editor_propose_edit", "memory_propose", "shell_cd"):
            if not path_in_workspace(target, workspace):
                request.reason = "outside workspace"
                return AuthResult.ask(request, reason="outside workspace")
        return AuthResult.allow("default")


def load_harness(project_root: Path) -> Harness:
    root = Path(project_root).resolve()
    return Harness(project_root=root, policy=load_policy(root))


def ensure_harness(state: object) -> Harness:
    current = getattr(state, "harness", None)
    if isinstance(current, Harness):
        return current
    root = getattr(state, "project_root", None)
    harness = load_harness(Path(root) if root is not None else Path.cwd())
    setattr(state, "harness", harness)
    return harness


def _is_gated(tool: str, target: Path | None, workspace: Path) -> bool:
    if tool in GATED_TOOLS:
        return True
    if tool == "shell_cd" and not path_in_workspace(target, workspace):
        return True
    return False


def _matches_deny(rules: list[str], tool: str, *, command: str, target: Path | None) -> bool:
    if command:
        return any_command_match(rules, tool, command)
    return any(rule_matches(rule, tool, command=command, target=target) for rule in rules)


def _matches_ask(rules: list[str], tool: str, *, command: str, target: Path | None) -> bool:
    if command:
        return any_command_match(rules, tool, command)
    return any(rule_matches(rule, tool, command=command, target=target) for rule in rules)


def _matches_allow(rules: list[str], tool: str, *, command: str, target: Path | None) -> bool:
    if command:
        return allow_command(rules, tool, command)
    return any(rule_matches(rule, tool, command=command, target=target) for rule in rules)


def _circuit_breaker(command: str) -> bool:
    parts = split_compound(command) or ([command] if command else [])
    for part in parts:
        for pattern in _CIRCUIT_PATTERNS:
            if pattern.search(part):
                return True
    return False


def _summary(tool: str, *, command: str, target: Path | None) -> str:
    if command:
        return command
    if target is not None:
        return str(target)
    return tool


def _format_rules(rules: list[str]) -> str:
    if not rules:
        return "(none)"
    return ", ".join(rules)
