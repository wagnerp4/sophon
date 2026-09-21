from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harness.approval import DECISION_DENY, DECISION_ONCE, DECISION_PERSIST, path_in_workspace
from harness.gate import Harness, load_harness
from harness.match import (
    allow_command,
    first_token,
    infer_persist_rules,
    specifier_matches,
    split_compound,
)
from harness.policy import Policy, append_allow_rules, load_policy


class MatchTests(unittest.TestCase):
    def test_split_compound_and_redirection(self) -> None:
        parts = split_compound("git status && npm test")
        self.assertEqual(parts, ["git status", "npm test"])
        parts = split_compound("git status 2>&1 | Select-Object -First 5")
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].startswith("git status"))
        self.assertEqual(first_token("& git.exe status"), "git")

    def test_git_allow_does_not_cover_compound_remove_item(self) -> None:
        rules = ["shell_exec(git *)"]
        self.assertTrue(allow_command(rules, "shell_exec", "git status"))
        self.assertFalse(
            allow_command(
                rules,
                "shell_exec",
                "git status && Remove-Item -Recurse -Force C:\\",
            )
        )

    def test_word_boundary_glob(self) -> None:
        self.assertTrue(specifier_matches("ls *", "ls -la"))
        self.assertFalse(specifier_matches("ls *", "lsof"))
        self.assertTrue(specifier_matches("ls*", "lsof"))

    def test_persist_prefix(self) -> None:
        rules = infer_persist_rules("shell_exec", command="uv run pytest")
        self.assertEqual(rules, ["shell_exec(uv *)"])
        rules = infer_persist_rules("shell_exec", command="git status && pytest -q")
        self.assertEqual(rules, ["shell_exec(git *)", "shell_exec(pytest *)"])


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.workspace = self.tmp / "proj"
        self.workspace.mkdir()
        self.harness = Harness(
            project_root=self.workspace,
            policy=Policy(workspace=".", mode="agent", allow=[], ask=[], deny=[]),
        )

    def test_shell_exec_asks_unless_allowlisted(self) -> None:
        decisions: list[str] = []

        def ask(request) -> str:
            decisions.append(request.tool)
            return DECISION_DENY

        result = self.harness.authorize(
            "shell_exec",
            {"command": "pytest -q"},
            cwd=self.workspace,
            ask=ask,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.permission, "ask:deny")
        self.assertEqual(decisions, ["shell_exec"])

        self.harness.policy.allow = ["shell_exec(pytest *)"]
        result = self.harness.authorize(
            "shell_exec",
            {"command": "pytest -q"},
            cwd=self.workspace,
            ask=ask,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.permission, "allow")
        self.assertEqual(len(decisions), 1)

    def test_once_does_not_persist(self) -> None:
        def ask(_request) -> str:
            return DECISION_ONCE

        first = self.harness.authorize(
            "shell_exec",
            {"command": "git status"},
            cwd=self.workspace,
            ask=ask,
        )
        self.assertTrue(first.allowed)
        self.assertEqual(first.permission, "ask:once")
        second = self.harness.authorize(
            "shell_exec",
            {"command": "git status"},
            cwd=self.workspace,
            ask=lambda _r: DECISION_DENY,
        )
        self.assertFalse(second.allowed)

    def test_persist_writes_local_yaml(self) -> None:
        def ask(_request) -> str:
            return DECISION_PERSIST

        result = self.harness.authorize(
            "shell_exec",
            {"command": "git status"},
            cwd=self.workspace,
            ask=ask,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.permission, "ask:persist")
        local = self.workspace / ".sophon" / "harness.local.yaml"
        self.assertTrue(local.is_file())
        text = local.read_text(encoding="utf-8")
        self.assertIn("shell_exec(git *)", text)
        again = self.harness.authorize(
            "shell_exec",
            {"command": "git log"},
            cwd=self.workspace,
            ask=lambda _r: DECISION_DENY,
        )
        self.assertTrue(again.allowed)

    def test_circuit_breaker_asks_even_if_allowlisted(self) -> None:
        self.harness.policy.allow = ["shell_exec"]
        asked = {"n": 0}

        def ask(_request) -> str:
            asked["n"] += 1
            return DECISION_DENY

        result = self.harness.authorize(
            "shell_exec",
            {"command": "rm -rf /"},
            cwd=self.workspace,
            ask=ask,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(asked["n"], 1)
        result = self.harness.authorize(
            "shell_exec",
            {"command": "Remove-Item -Recurse -Force C:\\"},
            cwd=self.workspace,
            ask=ask,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(asked["n"], 2)

    def test_outside_workspace_editor_asks(self) -> None:
        inside = self.workspace / "src" / "app.py"
        inside.parent.mkdir()
        inside.write_text("x = 1\n", encoding="utf-8")
        outside = self.tmp / "other.py"
        outside.write_text("y = 2\n", encoding="utf-8")
        asked = {"n": 0}

        def ask(_request) -> str:
            asked["n"] += 1
            return DECISION_DENY

        ok = self.harness.authorize(
            "editor_propose_edit",
            {"path": str(inside)},
            cwd=self.workspace,
            target=inside,
            ask=ask,
        )
        self.assertTrue(ok.allowed)
        self.assertEqual(asked["n"], 0)
        blocked = self.harness.authorize(
            "editor_propose_edit",
            {"path": str(outside)},
            cwd=self.workspace,
            target=outside,
            ask=ask,
        )
        self.assertFalse(blocked.allowed)
        self.assertEqual(asked["n"], 1)

    def test_plan_mode_denies_gated_tools(self) -> None:
        self.harness.set_mode("plan")
        result = self.harness.authorize(
            "shell_exec",
            {"command": "pytest -q"},
            cwd=self.workspace,
            ask=lambda _r: DECISION_ONCE,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.permission, "deny")

    def test_policy_file_edit_denied(self) -> None:
        policy_path = self.workspace / ".sophon" / "harness.yaml"
        policy_path.parent.mkdir()
        policy_path.write_text("workspace: .\n", encoding="utf-8")
        result = self.harness.authorize(
            "editor_propose_edit",
            {"path": str(policy_path)},
            cwd=self.workspace,
            target=policy_path,
            ask=lambda _r: DECISION_ONCE,
        )
        self.assertFalse(result.allowed)
        self.assertIn("policy", result.reason)

    def test_path_in_workspace(self) -> None:
        self.assertTrue(path_in_workspace(self.workspace / "a.py", self.workspace))
        self.assertFalse(path_in_workspace(self.tmp / "a.py", self.workspace))

    def test_load_policy_merge_local_allow(self) -> None:
        root = self.workspace
        (root / ".sophon").mkdir()
        (root / ".sophon" / "harness.yaml").write_text(
            "workspace: .\nmode: agent\npermissions:\n  allow: []\n  ask: []\n  deny: []\n",
            encoding="utf-8",
        )
        append_allow_rules(root / ".sophon" / "harness.local.yaml", ["shell_exec(uv *)"])
        policy = load_policy(root)
        self.assertIn("shell_exec(uv *)", policy.allow)
        loaded = load_harness(root)
        result = loaded.authorize(
            "shell_exec",
            {"command": "uv run pytest"},
            cwd=root,
            ask=lambda _r: DECISION_DENY,
        )
        self.assertTrue(result.allowed)


if __name__ == "__main__":
    unittest.main()
