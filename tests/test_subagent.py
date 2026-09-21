from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harness.gate import Harness
from harness.policy import Policy
from harness.subagent.depth import SubagentDepthError, resolve_child_depth
from harness.subagent.driver import run_in_process
from harness.subagent.filter import filter_child_tools, tool_schema_name
from harness.subagent.seed import completed_turn_seed
from harness.subagent.service import (
    SubagentActivationLimitError,
    SubagentModelOverrideError,
    SubagentService,
)
from harness.subagent.tools import execute_subagent_tool, spawn_tools_in_schema
from harness.subagent.types import SubagentRun, SubagentStartRequest, SubagentResult


def _harness(mode: str = "agent") -> Harness:
    tmp = Path(tempfile.mkdtemp())
    workspace = tmp / "proj"
    workspace.mkdir()
    return Harness(
        project_root=workspace,
        policy=Policy(workspace=".", mode=mode, allow=[], ask=[], deny=[]),
        mode_override=mode,
    )


def _parent(**kwargs: object) -> SimpleNamespace:
    state = SimpleNamespace(
        backend_id="lmstudio",
        server_model="google/gemma-4-12b-qat",
        processor="shared-processor",
        model="shared-weights",
        messages=[],
        delegation_depth=0,
        quiet=False,
        tool_max_rounds=None,
        harness=_harness("agent"),
        project_root=Path(tempfile.mkdtemp()),
        last_permission="",
    )
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


class _FakeCompletion:
    def __init__(self, text: str, tool_calls: list | None = None) -> None:
        self.text = text
        self.tool_calls = list(tool_calls or [])
        self.prompt_tokens = 3
        self.completion_tokens = 5
        self.finish_reason = "stop"


class DepthTests(unittest.TestCase):
    def test_zero_to_one_ok(self) -> None:
        self.assertEqual(resolve_child_depth(0, 1), 1)

    def test_one_to_two_raises(self) -> None:
        with self.assertRaises(SubagentDepthError):
            resolve_child_depth(1, 1)

    def test_service_rejects_nested_depth(self) -> None:
        service = SubagentService(max_depth=1, max_active=1)
        parent = _parent(delegation_depth=1)
        request = SubagentStartRequest(prompt="do work", description="nested")
        with self.assertRaises(SubagentDepthError):
            service.start("spawn", request, parent)


class SeedTests(unittest.TestCase):
    def test_spawn_seed_empty_and_fork_drops_in_flight(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "in-flight"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "partial"},
        ]
        self.assertEqual(completed_turn_seed([]), [])
        seed = completed_turn_seed(messages)
        self.assertEqual([item.get("content") for item in seed], ["sys", "first", "done"])


class FilterTests(unittest.TestCase):
    def test_child_schema_drops_spawn_and_research_writes(self) -> None:
        tools = [
            _tool("subagent"),
            _tool("subagent_fork"),
            _tool("shell_exec"),
            _tool("editor_propose_edit"),
            _tool("vault_read"),
        ]
        general = [tool_schema_name(item) for item in filter_child_tools(tools, "general")]
        research = [tool_schema_name(item) for item in filter_child_tools(tools, "research")]
        self.assertEqual(general, ["shell_exec", "editor_propose_edit", "vault_read"])
        self.assertEqual(research, ["vault_read"])


class SchemaAndAskTests(unittest.TestCase):
    def test_plan_and_chat_omit_spawn_tools(self) -> None:
        with patch.dict("os.environ", {"SOPHON_SUBAGENT_TOOLS": "1"}):
            agent = SimpleNamespace(delegation_depth=0, harness=_harness("agent"))
            plan = SimpleNamespace(delegation_depth=0, harness=_harness("plan"))
            chat = SimpleNamespace(delegation_depth=0, harness=_harness("chat"))
            child = SimpleNamespace(delegation_depth=1, harness=_harness("agent"))
            self.assertTrue(spawn_tools_in_schema(agent))
            self.assertFalse(spawn_tools_in_schema(plan))
            self.assertFalse(spawn_tools_in_schema(chat))
            self.assertFalse(spawn_tools_in_schema(child))

    def test_plan_mode_execute_denied(self) -> None:
        parent = _parent(harness=_harness("plan"))
        out = execute_subagent_tool(
            parent,
            "subagent",
            {"description": "label", "prompt": "task"},
        )
        self.assertIn("plan mode", out)

    def test_ask_none_denies_shell_exec(self) -> None:
        harness = _harness("agent")
        result = harness.authorize(
            "shell_exec",
            {"command": "pytest -q"},
            cwd=harness.workspace_path(),
            ask=None,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.permission, "ask:deny")


class ConcurrencyAndForkModelTests(unittest.TestCase):
    def test_second_concurrent_start_refused(self) -> None:
        service = SubagentService(max_depth=1, max_active=1)
        parent = _parent()
        request = SubagentStartRequest(prompt="one", description="first")
        nested = {"hit": False}

        def fake_run(*_args: object, **_kwargs: object) -> SubagentRun:
            nested["hit"] = True
            with self.assertRaises(SubagentActivationLimitError):
                service.start(
                    "spawn",
                    SubagentStartRequest(prompt="two", description="second"),
                    parent,
                )
            return SubagentRun(
                id="run1",
                provider="spawn",
                depth=1,
                agent_type="general",
                label="first",
                result=SubagentResult(output="ok"),
            )

        with patch("harness.subagent.driver.run_in_process", side_effect=fake_run):
            run = service.start("spawn", request, parent)
        self.assertTrue(nested["hit"])
        self.assertEqual(run.result.output, "ok")
        self.assertEqual(service._active, 0)

    def test_fork_cannot_change_model(self) -> None:
        service = SubagentService(max_depth=1, max_active=1)
        parent = _parent()
        request = SubagentStartRequest(
            prompt="continue",
            description="fork",
            agent_options={"model": "openai:gpt-4.1"},
        )
        with self.assertRaises(SubagentModelOverrideError):
            service.start("fork", request, parent)


class DriverTests(unittest.TestCase):
    def test_spawn_empty_fork_seed_and_research_filter(self) -> None:
        parent_messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "in-flight"},
        ]
        parent = _parent(messages=parent_messages)
        captured: dict[str, object] = {}

        def session_tools(state: object) -> tuple[list[dict], None]:
            captured["depth"] = getattr(state, "delegation_depth", None)
            captured["backend"] = getattr(state, "backend_id", None)
            captured["server_model"] = getattr(state, "server_model", None)
            captured["processor"] = getattr(state, "processor", None)
            captured["model"] = getattr(state, "model", None)
            captured["quiet"] = getattr(state, "quiet", None)
            return (
                [
                    _tool("subagent"),
                    _tool("subagent_fork"),
                    _tool("shell_exec"),
                    _tool("vault_read"),
                ],
                None,
            )

        def complete_turn(state: object, messages: list, tools: list | None):
            captured.setdefault("turns", []).append(
                {
                    "messages": [dict(item) for item in messages],
                    "tools": [tool_schema_name(item) for item in (tools or [])],
                    "agent_type": getattr(state, "subagent_agent_type", None),
                }
            )
            return _FakeCompletion("child-out")

        spawn = run_in_process(
            parent,
            SubagentStartRequest(prompt="standalone", description="spawn-job"),
            "spawn",
            1,
            complete_turn=complete_turn,
            session_tools=session_tools,
            execute_tool=lambda *_a, **_k: "unused",
        )
        spawn_turn = captured["turns"][0]
        spawn_roles = [(item.get("role"), item.get("content")) for item in spawn_turn["messages"]]
        self.assertEqual(spawn.seed_len, 0)
        self.assertEqual(spawn.result.output, "child-out")
        self.assertEqual(spawn.result.stop_reason, "completed")
        self.assertNotIn(("user", "first"), spawn_roles)
        self.assertIn(("user", "standalone"), spawn_roles)
        self.assertEqual(spawn_turn["tools"], ["shell_exec", "vault_read"])
        self.assertEqual(captured["backend"], "lmstudio")
        self.assertEqual(captured["server_model"], "google/gemma-4-12b-qat")
        self.assertIs(captured["processor"], "shared-processor")
        self.assertIs(captured["model"], "shared-weights")
        self.assertEqual(captured["depth"], 1)
        self.assertTrue(captured["quiet"])
        self.assertEqual(parent.messages, parent_messages)

        fork = run_in_process(
            parent,
            SubagentStartRequest(
                prompt="new instruction",
                description="fork-job",
                agent_type="research",
            ),
            "fork",
            1,
            complete_turn=complete_turn,
            session_tools=session_tools,
            execute_tool=lambda *_a, **_k: "unused",
        )
        fork_turn = captured["turns"][1]
        fork_contents = [item.get("content") for item in fork_turn["messages"]]
        self.assertEqual(fork.seed_len, 2)
        self.assertEqual(fork.result.output, "child-out")
        self.assertIn("first", fork_contents)
        self.assertIn("done", fork_contents)
        self.assertNotIn("in-flight", fork_contents)
        self.assertIn("new instruction", fork_contents)
        self.assertEqual(fork_turn["tools"], ["vault_read"])
        self.assertEqual(fork_turn["agent_type"], "research")


if __name__ == "__main__":
    unittest.main()
