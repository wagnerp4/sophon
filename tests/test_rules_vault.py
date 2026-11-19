from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harness.rules_vault import (
    CLAUDE_MD_CONTENT,
    GENERATED_MARKER,
    RulesError,
    clean_body,
    find_project,
    render_agents_md,
    sync,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class CleanBodyTests(unittest.TestCase):
    def test_title_links_comments_and_headings(self) -> None:
        body = (
            "# Git\n\nSee [[Global/Pipeline/gates|the gates]] and [[principles]].\n"
            "%% human only %%\n## Branches\n```text\n# not a heading\n```\n"
        )
        title, text = clean_body(body, shift=2)
        self.assertEqual(title, "Git")
        self.assertIn("See the gates and principles.", text)
        self.assertNotIn("human only", text)
        self.assertIn("#### Branches", text)
        self.assertIn("# not a heading", text)


class SyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.vault = base / "vault"
        self.repo = base / "repo"
        self.repo.mkdir()
        _write(
            self.vault / "Projects/demo/project.md",
            f"---\nagents: export\nlayer: project\nproject: demo\nrepo: {self.repo}\n"
            "languages: [python]\norder: 0\n---\n# demo\n\nDemo repo.\n",
        )
        _write(
            self.vault / "Projects/other/rules.md",
            "---\nagents: export\nlayer: project\nproject: other\n---\n# Other\n\nnope\n",
        )
        _write(
            self.vault / "Global/git.md",
            "---\nagents: export\nlayer: global\norder: 30\n---\n# Git\n\nUse branches.\n",
        )
        _write(
            self.vault / "Global/Languages/python.md",
            "---\nagents: export\nlayer: global\nlanguage: python\norder: 41\n---\n# Python\n\nuv.\n",
        )
        _write(
            self.vault / "Global/Languages/typescript.md",
            "---\nagents: export\nlayer: global\nlanguage: typescript\n---\n# TypeScript\n\nx\n",
        )
        _write(
            self.vault / "Global/draft.md",
            "---\nlayer: global\n---\n# Draft\n\nsecret plan\n",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_render_layers_and_filters(self) -> None:
        text = render_agents_md(self.vault, find_project(self.vault, self.repo))
        self.assertTrue(text.startswith(GENERATED_MARKER))
        self.assertLess(text.index("Demo repo."), text.index("Use branches."))
        self.assertIn("### Python", text)
        self.assertNotIn("TypeScript", text)
        self.assertNotIn("nope", text)
        self.assertNotIn("secret plan", text)

    def test_sync_then_check(self) -> None:
        first = sync(self.repo, self.vault)
        self.assertTrue(first.changed)
        self.assertTrue(first.claude_created)
        self.assertEqual(
            (self.repo / "CLAUDE.md").read_text(encoding="utf-8"), CLAUDE_MD_CONTENT
        )
        self.assertFalse(sync(self.repo, self.vault, check=True).changed)
        _write(
            self.vault / "Global/git.md",
            "---\nagents: export\nlayer: global\n---\n# Git\n\nnew\n",
        )
        drift = sync(self.repo, self.vault, check=True)
        self.assertTrue(drift.changed)
        self.assertNotIn("new", (self.repo / "AGENTS.md").read_text(encoding="utf-8"))

    def test_refuses_hand_written_agents_md(self) -> None:
        _write(self.repo / "AGENTS.md", "# my rules\n")
        with self.assertRaises(RulesError):
            sync(self.repo, self.vault)
        self.assertTrue(sync(self.repo, self.vault, force=True).changed)

    def test_unreadable_note_is_a_clean_error(self) -> None:
        _write(self.vault / "Global/bad.md", "")
        (self.vault / "Global/bad.md").write_bytes(
            b"---\nagents: export\n---\n\xff\xfe"
        )
        with self.assertRaises(RulesError) as ctx:
            sync(self.repo, self.vault)
        self.assertIn("Global/bad.md", str(ctx.exception))

    def test_unknown_repo(self) -> None:
        with self.assertRaises(RulesError):
            find_project(self.vault, self.repo.parent)


if __name__ == "__main__":
    unittest.main()
