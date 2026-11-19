"""sophon-011: safe keybind loader. Repo files are a validated, allow-listed overlay."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml

from cli.tui import keybinds as kb

PROTECTED = (
    "permission_once",
    "permission_persist",
    "permission_deny",
    "listen",
    "copy_chat",
    "open_models",
    "open_dashboard",
    "cycle_mode",
    "cycle_tree",
    "refresh_tree",
    "accept_edits",
    "energy_api",
    "quit",
)
UNKNOWN = ("google_connect", "search_mail", "web_search", "edit_keybinds")
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def _write(path: Path, text: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(
            self._tmp.cleanup
        )  # runs last: later cleanups may chmod inside it
        base = Path(self._tmp.name)
        self.profile = base / "profile"
        self.workspace = base / "ws"
        self.profile.mkdir()
        self.workspace.mkdir()
        patcher = mock.patch.object(kb, "sophon_project_root", lambda: self.profile)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"SOPHON_SKIP_DOTENV": "1"})
        env.start()
        self.addCleanup(env.stop)

    def profile_file(self, text: str) -> Path:
        return _write(self.profile / ".sophon" / "keybinds.yaml", text)

    def ws_file(self, text: str | bytes) -> Path:
        return _write(self.workspace / ".sophon" / "keybinds.yaml", text)

    def ignored(self, load: kb.KeybindLoad) -> dict[str, str]:
        return dict(load.ignored)


class SlashAndActionPolicyTests(_Base):
    def test_ac1_workspace_slash_entries_are_never_applied(self) -> None:
        self.ws_file(
            'slash:\n  system: ctrl+s\n  "rules trust 123456": f5\n  "assist accept": ctrl+o\n'
        )
        load = kb.load_effective_keybinds(self.workspace)
        self.assertEqual(load.binds.slash, kb.load_keybinds().slash)
        ignored = self.ignored(load)
        for name in ("system", "rules trust 123456", "assist accept"):
            entry = f"slash {kb._show('/' + name)}"
            self.assertEqual(
                ignored.get(entry), "slash commands are profile-only", entry
            )

    def test_ac2_only_allow_listed_actions_apply(self) -> None:
        lines = [
            f"{name}: ctrl+shift+{i % 10}" for i, name in enumerate(PROTECTED + UNKNOWN)
        ]
        lines.append("find_in_file: ctrl+alt+f")
        self.ws_file("\n".join(lines) + "\n")
        load = kb.load_effective_keybinds(self.workspace)
        profile = kb.load_keybinds()
        ignored = self.ignored(load)
        for name in PROTECTED:
            self.assertEqual(ignored.get(f"action {name}"), "protected action", name)
            self.assertEqual(
                load.binds.actions.get(name), profile.actions.get(name), name
            )
        for name in UNKNOWN:
            self.assertEqual(ignored.get(f"action {name}"), "unknown action", name)
            self.assertNotIn(name, load.binds.actions)
        self.assertEqual(load.binds.actions["find_in_file"], "ctrl+alt+f")

    def test_ac3_overlay_keeps_customised_profile(self) -> None:
        self.profile_file(
            "find_in_file: ctrl+shift+f\nsave_file: ctrl+s\n"
            "slash:\n  models: ctrl+shift+m\n  permissions: f2\n"
        )
        self.ws_file("save_file: ctrl+k\n")
        profile = kb.load_keybinds()
        load = kb.load_effective_keybinds(self.workspace)
        self.assertEqual(load.binds.actions["save_file"], "ctrl+k")
        self.assertEqual(load.binds.actions["find_in_file"], "ctrl+shift+f")
        for name, key in profile.actions.items():
            if name != "save_file":
                self.assertEqual(load.binds.actions[name], key, name)
        self.assertEqual(load.binds.slash, profile.slash)
        self.assertEqual(load.ignored, ())


class KeyValidationTests(_Base):
    BAD = ("ctrl+s,escape", "[/x]", "ctrl+ s", "a b", "ctrl+[b]", "k" * 30, "ctrl+s\n")

    def test_ac4_bad_keys_rejected_in_both_sources(self) -> None:
        for bad in self.BAD:
            with self.subTest(bad=bad):
                self.assertIsNone(kb.normalise_key(bad))
                quoted = yaml.safe_dump({"save_file": bad}).strip()
                self.profile_file(quoted + "\n")
                self.assertEqual(kb.load_keybinds().actions["save_file"], "ctrl+s")
                problems = kb.profile_keybind_problems()
                self.assertTrue(any("save_file" in p for p in problems), problems)
                self.ws_file(quoted + "\n")
                load = kb.load_effective_keybinds(self.workspace)
                self.assertEqual(load.binds.actions["save_file"], "ctrl+s")
                self.assertTrue(any(entry == "file" for entry, _ in load.ignored))

    def test_ac4_normalisation_keeps_key_case(self) -> None:
        self.assertEqual(kb.normalise_key("Ctrl+S "), "ctrl+S")
        self.assertEqual(kb.normalise_key("A"), "A")
        self.assertEqual(kb.normalise_key(""), "")

    def test_ac4_workspace_keys_need_modifier_or_fkey(self) -> None:
        for bare in ("escape", "enter", "1", "right"):
            with self.subTest(key=bare):
                self.ws_file(f"save_as: {bare}\n")
                load = kb.load_effective_keybinds(self.workspace)
                self.assertEqual(
                    self.ignored(load).get("action save_as"),
                    "key needs a modifier or F-key",
                )
        self.ws_file("accept_completion: right\n")  # equals the profile's own key
        self.assertEqual(kb.load_effective_keybinds(self.workspace).ignored, ())
        for good in ("ctrl+k", "f9"):
            self.ws_file(f"save_as: {good}\n")
            self.assertEqual(
                kb.load_effective_keybinds(self.workspace).binds.actions["save_as"],
                good,
            )

    def test_ac4_collisions_and_reserved_keys(self) -> None:
        self.ws_file("save_as: f2\nnew_file: ctrl+m\n")
        ignored = self.ignored(kb.load_effective_keybinds(self.workspace))
        self.assertEqual(ignored["action save_as"], "key already bound")
        self.assertEqual(ignored["action new_file"], "key already bound")
        self.ws_file("save_as: ctrl+k\nnew_file: ctrl+k\n")
        load = kb.load_effective_keybinds(self.workspace)
        self.assertEqual(load.binds.actions["save_as"], "ctrl+k")
        self.assertEqual(self.ignored(load)["action new_file"], "key already bound")
        for reserved in (
            "ctrl+q",
            "ctrl+c",
            "ctrl+p",
            "tab",
            "shift+tab",
            "f1",
            "super+c",
            "Ctrl+Q",
        ):
            with self.subTest(key=reserved):
                self.ws_file(f"save_as: {reserved}\n")
                load = kb.load_effective_keybinds(self.workspace)
                self.assertEqual(
                    self.ignored(load).get("action save_as"), "reserved key"
                )
                for entry, reason in load.ignored:
                    self.assertTrue(entry and reason)


class ProfileSlashAndSetKeybindTests(_Base):
    def test_ac5_profile_slash_names(self) -> None:
        long_name = "x" * 129
        self.profile_file(
            yaml.safe_dump(
                {
                    "slash": {
                        "tts on": "f5",
                        "models": "ctrl+shift+m",
                        "permissions": "f2",
                        "x')": "f7",
                        'a"b': "f8",
                        "a\\b": "f10",
                        "x\n": "f11",
                        "": "f12",
                        long_name: "f13",
                    }
                }
            )
        )
        slash = kb.load_keybinds().slash
        self.assertEqual(slash["tts on"], "f5")
        self.assertIn("models", slash)
        self.assertIn("permissions", slash)
        for bad in ("x')", 'a"b', "a\\b", "x\n", "x", "", long_name):
            self.assertNotIn(bad, slash)

    def test_ac5_set_keybind_preserves_entries_and_header(self) -> None:
        path = self.profile_file(
            'save_file: ctrl+s\nslash:\n  "x\')": f7\n  "tts on": f5\n'
        )
        self.assertEqual(kb.set_keybind("save_file", "ctrl+alt+o"), "")
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# sophon keybinds"))
        data = yaml.load(text, Loader=kb._NoAliasLoader)
        self.assertEqual(data["save_file"], "ctrl+alt+o")
        self.assertEqual(data["slash"], {"x')": "f7", "tts on": "f5"})

    def test_ac5_set_keybind_refuses_invalid_key_and_unsafe_files(self) -> None:
        path = self.profile_file("save_file: ctrl+s\n")
        before = path.read_bytes()
        self.assertTrue(
            kb.set_keybind("save_file", "ctrl+s,f9").startswith("error: invalid key")
        )
        self.assertEqual(path.read_bytes(), before)
        for broken in ("a: &x b\nc: *x\n", "save_file: [unclosed\n", "- a\n- b\n"):
            with self.subTest(broken=broken):
                path = self.profile_file(broken)
                before = path.read_bytes()
                message = kb.set_keybind("find_in_file", "ctrl+alt+f")
                self.assertTrue(message.startswith("error:"), message)
                self.assertEqual(path.read_bytes(), before)

    def test_ac5_unquoted_int_loads_as_string(self) -> None:
        self.profile_file("permission_once: 4\n")
        self.assertEqual(kb.load_keybinds().actions["permission_once"], "4")

    def test_amendment_empty_profile_is_editable_and_template_never_overwrites(
        self,
    ) -> None:
        path = self.profile_file("# only a comment\n")
        self.assertEqual(kb.set_keybind("save_file", "ctrl+alt+o"), "")
        self.assertEqual(
            yaml.safe_load(path.read_text(encoding="utf-8"))["save_file"], "ctrl+alt+o"
        )
        before = path.read_bytes()
        kb.ensure_keybinds_file(self.profile)
        self.assertEqual(path.read_bytes(), before)

    def test_amendment_set_keybind_refuses_oversized_result(self) -> None:
        lines, size, i = [], 0, 0
        while True:  # whole lines only, so the file stays valid YAML just under the cap
            line = f"unknown_{i}: ctrl+k\n"
            if size + len(line) > kb.MAX_KEYBINDS_BYTES - 40:
                break
            lines.append(line)
            size += len(line)
            i += 1
        path = self.profile_file("".join(lines))
        self.assertEqual(kb.read_keybinds_file(path).problems, ())
        before = path.read_bytes()
        message = kb.set_keybind("save_file", "ctrl+alt+o")
        self.assertIn("would exceed 64 KiB", message)
        self.assertEqual(path.read_bytes(), before)


class HostileFileTests(_Base):
    def problems_for(self, content: str | bytes) -> tuple[str, ...]:
        path = self.ws_file(content)
        start = time.monotonic()
        raw = kb.read_keybinds_file(path)
        self.elapsed = time.monotonic() - start
        return raw.problems

    def test_ac6_oversized_and_symlinked_oversized(self) -> None:
        self.assertEqual(
            self.problems_for("a" * (kb.MAX_KEYBINDS_BYTES + 1)),
            ("file larger than 64 KiB",),
        )
        big = _write(self.profile / "big.yaml", "a" * (kb.MAX_KEYBINDS_BYTES + 5))
        link = self.workspace / ".sophon" / "keybinds.yaml"
        link.unlink()
        link.symlink_to(big)
        self.assertEqual(
            kb.read_keybinds_file(link).problems, ("file larger than 64 KiB",)
        )

    def test_ac6_alias_bomb_rejected_fast(self) -> None:
        bomb = 'a: &a ["x","x","x","x","x","x","x","x","x"]\n'
        for level in range(ord("b"), ord("j")):
            prev = chr(level - 1)
            bomb += f"{chr(level)}: &{chr(level)} [*{prev},*{prev},*{prev},*{prev},*{prev},*{prev},*{prev},*{prev},*{prev}]\n"
        bomb += "#" * max(0, 466 - len(bomb)) + "\n"
        self.assertGreaterEqual(len(bomb.encode()), 466)
        self.assertEqual(self.problems_for(bomb), ("aliases are not allowed",))
        self.assertLess(self.elapsed, 0.1)

    def test_ac6_deep_nesting_bounded(self) -> None:
        problems = self.problems_for("[" * (kb.MAX_KEYBINDS_BYTES - 1))
        self.assertEqual(len(problems), 1)
        self.assertLess(self.elapsed, 3.0)

    def test_ac6_typed_scalars_never_raise(self) -> None:
        for doc in (
            "save_file: 2020-13-45\n",
            "save_file: !!timestamp hello\n",
            "save_file: !!int hello\n",
            "save_file: !!float hello\n",
            "save_file: !!bool hello\n",
            "save_file: " + "9" * 5000 + "\n",
            "9" * 5000 + ": ctrl+k\n",
            '"\\U00110000": ctrl+k\n',
            'a: "\\UFFFFFFFF"\n',
        ):
            with self.subTest(doc=doc[:30]):
                raw = kb.read_keybinds_file(self.ws_file(doc))
                values = list(raw.actions.values()) + list(raw.actions)
                self.assertTrue(raw.problems or values, doc[:30])
                for item in values:
                    self.assertIsInstance(item, str)
        # typed-looking scalars load as plain strings and are then validated as keys
        raw = kb.read_keybinds_file(self.ws_file("save_file: 2020-13-45\n"))
        self.assertEqual(raw.actions, {})
        self.assertTrue(raw.problems[0].startswith("action save_file: invalid key"))

    def test_ac6_problem_list_is_capped(self) -> None:
        doc = "".join(f"a{i}: [x]\n" for i in range(6000))
        problems = self.problems_for(doc)
        self.assertEqual(len(problems), kb.MAX_REPORTED + 1)
        self.assertTrue(problems[-1].startswith("and "))

    def test_ac6_misc_bad_shapes(self) -> None:
        self.assertTrue(
            self.problems_for("save_file: [unclosed\n")[0].startswith("invalid YAML")
        )
        self.assertEqual(self.problems_for(b"save_file: \xff\n"), ("not UTF-8",))
        self.assertEqual(
            self.problems_for("- a\n- b\n"), ("top level is not a mapping",)
        )
        problems = self.problems_for("save_file: [ctrl+s]\nfind_in_file: {a: b}\n")
        self.assertEqual(len(problems), 2)
        raw = kb.read_keybinds_file(
            self.ws_file("save_file: 7\nnew_file: true\n42: ctrl+k\n")
        )
        self.assertEqual(
            raw.actions, {"save_file": "7", "new_file": "true", "42": "ctrl+k"}
        )
        load = kb.load_effective_keybinds(self.workspace)
        self.assertEqual(self.ignored(load).get("action 42"), "unknown action")
        self.assertEqual(
            self.problems_for("slash: [a, b]\n"), ("slash: not a mapping",)
        )

    @unittest.skipIf(IS_ROOT or not hasattr(os, "symlink"), "needs non-root POSIX")
    def test_amendment_unreadable_symlink_targets(self) -> None:
        locked_file = _write(self.profile / "locked.yaml", "save_file: ctrl+k\n")
        locked_file.chmod(0)
        locked_dir = self.profile / "locked_dir"
        _write(locked_dir / "f.yaml", "save_file: ctrl+k\n")
        locked_dir.chmod(0)
        self.addCleanup(locked_dir.chmod, 0o755)
        self.addCleanup(locked_file.chmod, 0o644)
        link = self.workspace / ".sophon" / "keybinds.yaml"
        link.parent.mkdir(parents=True, exist_ok=True)
        for target in (locked_file, locked_dir / "f.yaml"):
            with self.subTest(target=target.name):
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(target)
                raw = kb.read_keybinds_file(link)
                self.assertTrue(raw.problems[0].startswith("unreadable"), raw.problems)
                load = kb.load_effective_keybinds(self.workspace)
                self.assertEqual(load.binds.actions, kb.load_keybinds().actions)
                self.assertTrue(load.ignored[0][1].startswith("unreadable"))

    def test_amendment_loader_is_pure_python_base_loader(self) -> None:
        self.assertTrue(issubclass(kb._NoAliasLoader, yaml.BaseLoader))
        cloader = getattr(yaml, "CBaseLoader", None)
        if cloader is not None:
            self.assertFalse(issubclass(kb._NoAliasLoader, cloader))

    def test_amendment_messages_escape_file_content(self) -> None:
        self.ws_file('save_file: "\\e]52;c;aGk=\\a"\n"\\e[31m": ctrl+k\n')
        load = kb.load_effective_keybinds(self.workspace)
        for entry, reason in load.ignored:
            for text in (entry, reason):
                self.assertNotIn("\x1b", text)
                self.assertNotIn("\x07", text)
                self.assertLessEqual(len(text), len("action ") + 80)
        self.assertLessEqual(len(kb._show("x" * 500)), 80)


class NoWriteTests(_Base):
    def test_ac7_workspace_loads_never_write(self) -> None:
        other = Path(self._tmp.name) / "other"
        other.mkdir()
        kb.load_effective_keybinds(self.workspace)
        kb.read_keybinds_file(self.workspace / "missing.yaml")
        kb.load_keybinds(other)
        self.assertEqual(list(self.workspace.iterdir()), [])
        self.assertEqual(list(other.iterdir()), [])

    def test_ac7_profile_template_still_created(self) -> None:
        kb.load_keybinds()
        self.assertTrue((self.profile / ".sophon" / "keybinds.yaml").is_file())

    def test_ac7_workspace_symlink_to_profile_root_is_profile(self) -> None:
        self.profile_file("save_file: ctrl+s\nslash:\n  models: ctrl+shift+m\n")
        alias_root = Path(self._tmp.name) / "alias"
        alias_root.symlink_to(self.profile)
        unresolved = Path(self._tmp.name) / "." / "profile"
        with mock.patch.object(kb, "sophon_project_root", lambda: unresolved):
            load = kb.load_effective_keybinds(alias_root)
        self.assertIsNone(load.source)
        self.assertEqual(load.ignored, ())

    def test_ac7_workspace_file_symlinked_to_profile_file_uses_workspace_policy(
        self,
    ) -> None:
        target = self.profile_file(
            "listen: ctrl+l\nsave_as: ctrl+alt+s\nslash:\n  system: f6\n"
        )
        link = self.workspace / ".sophon" / "keybinds.yaml"
        link.parent.mkdir(parents=True)
        link.symlink_to(target)
        load = kb.load_effective_keybinds(self.workspace)
        ignored = self.ignored(load)
        self.assertEqual(ignored.get("action listen"), "protected action")
        self.assertIn(f"slash {kb._show('/system')}", ignored)


class SurfaceTests(_Base):
    def test_ac8_permission_and_nav_keys_unaffected(self) -> None:
        self.ws_file(
            "permission_once: x\nopen_dashboard: ctrl+k\nslash:\n  system: f6\n"
        )
        profile = kb.load_keybinds()
        effective = kb.load_effective_keybinds(self.workspace).binds
        self.assertEqual(kb.permission_keys(effective), kb.permission_keys(profile))
        self.assertEqual(kb.nav_chord_keys(effective), kb.nav_chord_keys(profile))

    def test_ac10_nav_keys_of_effective_set_equal_profile(self) -> None:
        self.ws_file("open_dashboard: x\n")
        profile = kb.load_keybinds()
        effective = kb.load_effective_keybinds(self.workspace).binds
        for action in kb.NAV_ACTIONS:
            self.assertEqual(effective.actions[action], profile.actions[action])


class EditorWiringTests(unittest.TestCase):
    def test_ac9_editor_uses_effective_loader(self) -> None:
        from cli.tui.screens import (
            editor,  # requires the tui extra; fails loudly without it
        )

        root = Path(tempfile.gettempdir()) / "sophon-011-ws"
        expected = kb.KeybindLoad(
            kb.KeybindSet(actions={"save_file": "ctrl+k"}, slash={"models": "f3"}),
            (("action listen", "protected action"),),
            root / ".sophon" / "keybinds.yaml",
        )
        stub = mock.Mock()
        stub._root = root
        with mock.patch.object(
            editor, "load_effective_keybinds", return_value=expected
        ) as spy:
            editor.EditorScreen._load_keybinds(stub)
        spy.assert_called_once_with(root)
        self.assertEqual(stub._keybinds, {"save_file": "ctrl+k"})
        self.assertEqual(stub._slash_binds, {"models": "f3"})
        self.assertEqual(stub._keybinds_ignored, expected.ignored)


class WriteModeTests(_Base):
    @unittest.skipIf(os.name != "posix", "POSIX file modes")
    def test_set_keybind_keeps_file_mode(self) -> None:
        path = self.profile_file("save_file: ctrl+s\n")
        path.chmod(0o644)
        self.assertEqual(kb.set_keybind("save_file", "ctrl+alt+o"), "")
        self.assertEqual(path.stat().st_mode & 0o777, 0o644)

    @unittest.skipIf(os.name != "posix" or IS_ROOT, "POSIX modes, non-root")
    def test_set_keybind_refuses_read_only_profile(self) -> None:
        path = self.profile_file("save_file: ctrl+s\n")
        path.chmod(0o444)
        self.addCleanup(path.chmod, 0o644)
        before = path.read_bytes()
        message = kb.set_keybind("save_file", "ctrl+alt+o")
        self.assertTrue(message.startswith("error: could not write"), message)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob(".keybinds.*")), [])


if __name__ == "__main__":
    unittest.main()
