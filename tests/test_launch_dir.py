"""sophon-010: sophon knows the launch directory on every spawn route."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cli.host import launch
from utils.device import platform as plat
from utils.device.launch_token import TOKEN_ENV, TOKEN_OPTION

PRESET = "gemma4_12b_it"
HOSTILE = "R&D x;y%P%[/b] '\"$`!()^"


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        launch._reset_for_tests()
        self.addCleanup(launch._reset_for_tests)
        env = mock.patch.dict(os.environ, {"SOPHON_SKIP_DOTENV": "1"})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(TOKEN_ENV, None)
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(self.tmp)

    def make_dir(self, name: str) -> Path:
        path = self.tmp / name
        path.mkdir(parents=True)
        return path.resolve()


class TokenTests(_Base):
    def test_ac2_round_trip_and_alphabet(self) -> None:
        names = [
            "with space",
            "& ^ % ; ! ( ) ' \" $ `",
            "[/x]",
            "ünïcødé",
            "\\\\wsl.localhost\\D\\x",
        ]
        for name in names:
            with self.subTest(name=name):
                token = launch.encode_token(name)
                self.assertRegex(token, r"^[A-Za-z0-9_-]+$")
                self.assertFalse(token.startswith("-"))
                self.assertEqual(launch.decode_token(token), Path(name))
        raw = os.fsdecode(
            b"/tmp/\xff"
        )  # a non-UTF-8 Linux name round-trips via surrogates
        self.assertEqual(launch.decode_token(launch.encode_token(raw)), Path(raw))
        absolute = launch.encode_token("/any/abs/path")
        self.assertFalse(absolute.startswith("-"))

    def test_decode_rejects_malformed(self) -> None:
        for bad in ("", "!!!", "a+b", "a/b", "a=", "x" * 8193, "AA AA"):
            with self.subTest(bad=bad[:20]):
                self.assertIsNone(launch.decode_token(bad))
        self.assertIsNone(launch.decode_token(launch.encode_token("a\x00b")))

    def test_ac3_with_token_arg_and_token_from_argv(self) -> None:
        path = self.tmp
        argv = [
            "chat",
            "--preset",
            PRESET,
            TOKEN_OPTION,
            "AAA",
            f"{TOKEN_OPTION}=BBB",
            "--x",
            TOKEN_OPTION,
        ]
        out = launch.with_token_arg(argv, path)
        self.assertEqual(out[:4], ["chat", "--preset", PRESET, "--x"])
        self.assertEqual(out.count(TOKEN_OPTION), 1)
        self.assertEqual(out[-2], TOKEN_OPTION)
        self.assertEqual(launch.decode_token(out[-1]), path)
        self.assertEqual(
            launch.token_from_argv(["a", TOKEN_OPTION, "X", f"{TOKEN_OPTION}=Y"]), "Y"
        )
        self.assertIsNone(launch.token_from_argv(["a", TOKEN_OPTION]))
        with mock.patch.object(launch, "encode_token", return_value=None):
            self.assertEqual(
                launch.with_token_arg(["chat", TOKEN_OPTION, "Z"], path), ["chat"]
            )


class RecordTests(_Base):
    def test_ac1_argv_beats_env_beats_cwd_and_env_is_popped(self) -> None:
        arg_dir = self.make_dir("arg")
        env_dir = self.make_dir("env")
        os.environ[TOKEN_ENV] = launch.encode_token(env_dir)
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(launch, "_is_windows_system_dir", return_value=False),
        ):
            self.assertEqual(
                launch.record_launch_dir(launch.encode_token(arg_dir)), arg_dir
            )
        self.assertNotIn(TOKEN_ENV, os.environ)
        launch._reset_for_tests()
        os.environ[TOKEN_ENV] = launch.encode_token(env_dir)
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(launch, "_is_windows_system_dir", return_value=False),
        ):
            self.assertEqual(launch.record_launch_dir(None), env_dir)
        launch._reset_for_tests()

    def test_ac1_env_token_ignored_on_linux(self) -> None:
        os.environ[TOKEN_ENV] = launch.encode_token(self.make_dir("env"))
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(launch.record_launch_dir(None), self.tmp)
        self.assertNotIn(TOKEN_ENV, os.environ)

    def test_ac1_invalid_tokens_warn_once_and_fall_through(self) -> None:
        a_file = self.tmp / "f.txt"
        a_file.write_text("x", encoding="utf-8")
        cases = [
            "!!!",
            "x" * 8193,
            "A",  # bad padding / length
            launch.encode_token("a\x00b"),
            launch.encode_token("relative/dir"),
            launch.encode_token(str(self.tmp / "missing")),
            launch.encode_token(str(a_file)),
        ]
        for token in cases:
            with self.subTest(token=token[:20]):
                launch._reset_for_tests()
                with mock.patch("sys.stderr") as err:
                    self.assertEqual(launch.record_launch_dir(token), self.tmp)
                messages = [
                    c.args[0]
                    for c in err.write.call_args_list
                    if "[sophon]" in c.args[0]
                ]
                self.assertEqual(len(messages), 1)
        launch._reset_for_tests()
        with (
            mock.patch.object(
                Path, "is_dir", side_effect=PermissionError(13, "denied")
            ),
            mock.patch("sys.stderr"),
        ):
            result = launch.record_launch_dir(launch.encode_token(str(self.tmp)))
        self.assertEqual(result, self.tmp)

    def test_ac1_windows_system_dir_falls_back_to_install_root(self) -> None:
        install = self.make_dir("install")
        with mock.patch.dict(os.environ, {"SystemRoot": "C:\\Windows"}):
            self.assertTrue(launch._is_windows_system_dir(Path("C:\\WINDOWS\\")))
            self.assertFalse(launch._is_windows_system_dir(Path("C:\\Users")))
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(launch, "_is_windows_system_dir", return_value=True),
            mock.patch(
                "utils.device.env_bootstrap.sophon_project_root", return_value=install
            ),
        ):
            self.assertEqual(launch.record_launch_dir(None), install)

    def test_ac1_first_value_sticks(self) -> None:
        first = self.make_dir("first")
        self.assertEqual(launch.record_launch_dir(launch.encode_token(first)), first)
        second = self.make_dir("second")
        self.assertEqual(launch.record_launch_dir(launch.encode_token(second)), first)
        self.assertEqual(launch.launch_dir(), first)

    def test_ac1_parse_chat_argv_survives_bad_token(self) -> None:
        from cli.terminal import parse_chat_argv

        with mock.patch("sys.stderr"):
            parse_chat_argv(["--no-spawn-window", TOKEN_OPTION, "!!!"])
        self.assertEqual(launch.launch_dir(), self.tmp)

    def test_parse_chat_argv_records_token_and_option_is_hidden(self) -> None:
        from click.testing import CliRunner

        from cli.terminal import chat_command, parse_chat_argv

        target = self.make_dir("repo")
        parse_chat_argv(
            ["--no-spawn-window", f"{TOKEN_OPTION}={launch.encode_token(target)}"]
        )
        self.assertEqual(launch.launch_dir(), target)
        help_text = CliRunner().invoke(chat_command, ["--help"]).output
        self.assertNotIn("launch-dir-token", help_text)


class SpawnTests(_Base):
    def _launch(self, argv: list[str]) -> list[str]:
        from cli.host import spawn

        captured: dict[str, list[str]] = {}

        def fake_spawn(command, env, cwd, **_kw):
            captured["command"] = list(command)
            return mock.Mock(pid=1)

        with (
            mock.patch.object(spawn, "spawn_desktop_terminal", side_effect=fake_spawn),
            mock.patch.object(spawn, "tail_log_until_exit", return_value=0),
            mock.patch.object(
                spawn, "make_session_log_path", return_value=self.tmp / "log" / "s.log"
            ),
            mock.patch.object(spawn, "install_windows_terminal_profile") as installer,
            mock.patch.object(spawn, "windows_profile_installed", return_value=True),
        ):
            self.assertEqual(spawn.launch_tui_in_new_terminal(argv), 0)
        installer.assert_not_called()
        return captured["command"]

    def test_ac4a_forwards_cwd_token_and_explicit_token_wins(self) -> None:
        command = self._launch(["chat", "--preset", PRESET])
        self.assertIn(PRESET, command)
        self.assertEqual(command.count(TOKEN_OPTION), 1)
        self.assertEqual(
            launch.decode_token(command[command.index(TOKEN_OPTION) + 1]), self.tmp
        )
        launch._reset_for_tests()
        other = self.make_dir("other")
        command = self._launch(["chat", TOKEN_OPTION, launch.encode_token(other)])
        self.assertEqual(command.count(TOKEN_OPTION), 1)
        self.assertEqual(
            launch.decode_token(command[command.index(TOKEN_OPTION) + 1]), other
        )

    def test_ac4a_unencodable_cwd_forwards_no_token(self) -> None:
        with (
            mock.patch.object(launch, "encode_token", return_value=None),
            mock.patch("sys.stderr"),
        ):
            command = self._launch(["chat"])
        self.assertNotIn(TOKEN_OPTION, command)

    def test_ac4b_raw_path_never_on_any_command_line(self) -> None:
        from cli.host import spawn

        hostile = self.make_dir(HOSTILE)
        child = launch.with_token_arg(["--no-spawn-window"], hostile)
        token = child[-1]
        env = {"SOPHON_TUI_LOG": "/x/log", "SOPHON_TUI_CHILD": "1"}
        popen = mock.Mock(return_value=mock.Mock(pid=2))

        def which_only(name: str):
            return {"wt": "/fake/wt", "konsole": "/fake/konsole"}.get(name)

        with (
            mock.patch.object(spawn.shutil, "which", side_effect=which_only),
            mock.patch.object(spawn.subprocess, "Popen", popen),
            mock.patch.object(spawn, "windows_profile_installed", return_value=True),
            mock.patch.object(spawn, "profile_name", return_value="sophon chat"),
        ):
            spawn._spawn_windows(
                ["sophon-chat-tui", *child],
                env,
                "/install",
                window_title="sophon chat",
                use_sophon_profile=True,
                utf8_console=True,
            )
            spawn._spawn_native_linux(
                ["sophon-chat-tui", *child], env, "/install", window_title="sophon chat"
            )
        for call in popen.call_args_list:
            joined = " ".join(call.args[0])
            self.assertIn(token, joined)
            for fragment in ("R&D", "x;y", "%P%", "[/b]", HOSTILE):
                self.assertNotIn(fragment, joined)


class WslTests(_Base):
    def test_ac5_env_token_via_wslenv(self) -> None:
        unc = "\\\\wsl.localhost\\D\\tmp\\x"

        def to_windows(path):
            return "C:\\deploy\\sophon-cli.exe" if str(path).endswith(".exe") else unc

        calls: list[dict] = []

        def fake_call(argv, env=None):
            calls.append({"argv": argv, "env": env})
            return 0

        before = dict(os.environ)
        with (
            mock.patch.dict(os.environ, {"WSLENV": "FOO/p:BAR"}),
            mock.patch.object(plat, "is_wsl", return_value=True),
            mock.patch.object(
                plat,
                "resolve_windows_console_script",
                return_value=Path("/mnt/c/deploy/sophon-cli.exe"),
            ),
            mock.patch.object(plat, "linux_path_to_windows", side_effect=to_windows),
            mock.patch.object(
                plat.shutil,
                "which",
                side_effect=lambda name: (
                    "/mnt/c/cmd.exe" if name == "cmd.exe" else None
                ),
            ),
            mock.patch.object(plat.subprocess, "call", side_effect=fake_call),
            mock.patch.object(sys, "platform", "linux"),
        ):
            self.assertEqual(
                plat.wsl_reexec_windows_cli(
                    "sophon-cli", ["--preset", PRESET], launch_dir=self.tmp
                ),
                0,
            )
            env = calls[-1]["env"]
            self.assertEqual(launch.decode_token(env[TOKEN_ENV]), Path(unc))
            self.assertEqual(env["WSLENV"].split(":"), ["FOO/p", "BAR", TOKEN_ENV])
            self.assertIn(f"--preset {PRESET}", calls[-1]["argv"][-1])
            self.assertNotIn(TOKEN_OPTION, calls[-1]["argv"][-1])
            self.assertNotIn(TOKEN_ENV, os.environ)
            # dir conversion fails -> no token, one warning
            with (
                mock.patch.object(
                    plat,
                    "linux_path_to_windows",
                    side_effect=lambda p: (
                        None if Path(p) == self.tmp else to_windows(p)
                    ),
                ),
                mock.patch("sys.stderr") as err,
            ):
                plat.wsl_reexec_windows_cli("sophon-cli", [], launch_dir=self.tmp)
            self.assertNotIn(TOKEN_ENV, calls[-1]["env"])
            warned = [
                c.args[0] for c in err.write.call_args_list if "[sophon]" in c.args[0]
            ]
            self.assertEqual(len(warned), 1)
            plat.wsl_reexec_windows_cli("sophon-cli", [], launch_dir=None)
            self.assertNotIn(TOKEN_ENV, calls[-1]["env"])
        self.assertEqual(
            {k: v for k, v in os.environ.items() if k != "SOPHON_SKIP_DOTENV"},
            {k: v for k, v in before.items() if k != "SOPHON_SKIP_DOTENV"},
        )

    def test_ac6_wslpath_timeout(self) -> None:
        seen: dict = {}

        def fake_run(argv, **kwargs):
            seen.update(kwargs)
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(plat, "is_wsl", return_value=True),
            mock.patch.object(plat.shutil, "which", return_value="/usr/bin/wslpath"),
            mock.patch.object(plat.subprocess, "run", side_effect=fake_run),
        ):
            self.assertIsNone(plat.linux_path_to_windows("/home/x"))
        self.assertIn("timeout", seen)


class TuiTests(_Base):
    def _app(self):
        from cli.terminal import parse_chat_argv
        from cli.tui.app import SophonTuiApp

        return SophonTuiApp(parse_chat_argv(["--no-spawn-window"]), None)

    def test_ac7_app_cwd_is_launch_dir(self) -> None:
        target = self.make_dir("repo")
        launch.record_launch_dir(launch.encode_token(target))
        self.assertEqual(self._app().cwd, target)

    def test_ac8_install_root_readers(self) -> None:
        from cli.tui import app as app_mod

        install = self.make_dir("install")
        app = mock.Mock()
        app.session_state = None
        app.params.preset = None
        app.params.model = None
        app.cwd = self.tmp
        seen: dict = {}

        class Stop(Exception):
            pass

        def startup(**kwargs):
            seen["startup"] = kwargs["cwd"]
            return PRESET, install / "m"

        def entries(root, **_kw):
            seen["entries"] = root
            raise Stop

        with (
            mock.patch.object(app_mod, "sophon_project_root", return_value=install),
            mock.patch.object(
                app_mod, "resolve_chat_startup_model", side_effect=startup
            ),
            mock.patch.object(
                app_mod, "build_model_picker_entries_for_root", side_effect=entries
            ),
            self.assertRaises(Stop),
        ):
            app_mod.ChatScreen._model_picker_entries(mock.Mock(), app)
        self.assertEqual(seen, {"startup": install, "entries": install})

        def preset_dir(key, root):
            seen["download"] = root
            raise Stop

        tui_app = self._app()
        tui_app.cwd = self.tmp
        screen = mock.Mock()
        screen.app = tui_app
        old_tqdm = os.environ.get("TQDM_POSITION")
        with (
            mock.patch.object(app_mod, "sophon_project_root", return_value=install),
            mock.patch.object(app_mod, "resolve_preset_dir", side_effect=preset_dir),
            mock.patch.object(tui_app, "call_from_thread"),
            mock.patch.object(tui_app, "post_message"),
        ):
            try:
                app_mod.ChatScreen._download_model_worker(screen, PRESET)
            except Stop:
                pass
        if old_tqdm is None:
            os.environ.pop("TQDM_POSITION", None)
        else:
            os.environ["TQDM_POSITION"] = old_tqdm
        self.assertEqual(seen["download"], install)

    def test_ac11_editor_mount_never_writes_into_launch_dir(self) -> None:
        from cli.tui.screens import editor

        install = self.make_dir("install")

        class Stop(Exception):
            pass

        stub = mock.Mock()
        stub._root = self.tmp
        stub._profile_root.return_value = install
        with (
            mock.patch.object(
                editor, "ensure_highlights_file", side_effect=Stop
            ) as spy,
            self.assertRaises(Stop),
        ):
            editor.EditorScreen.on_mount(stub)
        spy.assert_called_once_with(install)
        self.assertEqual(list(self.tmp.iterdir()), [install])

    def test_ac14_hostile_names_render_literally(self) -> None:
        from rich.markup import render

        from cli.tui.screens import editor
        from cli.tui.tiles import host

        name = "a[b]c[/x]\n"
        shown = launch.markup_safe_path(self.tmp / name)
        self.assertEqual(render(shown).plain, str(self.tmp / "a[b]c[/x]\\x0a"))
        stub = mock.Mock()
        stub._tree_kind = "project"
        component = "a[b]c[bold]x\n"  # one path component: POSIX names cannot hold "/"
        stub._tree_root.return_value = self.tmp / component
        title = editor.EditorScreen._tree_title(stub)
        self.assertEqual(render(title).plain, "Project · a[b]c[bold]x\\x0a")
        app = mock.Mock()
        app.cwd = self.tmp / name
        app.model_status_text = "m"
        snapshot = {"python": "3", "torch": "x", "host_memory": {}, "devices": []}
        with mock.patch(
            "utils.device.system_check.collect_system_snapshot", return_value=snapshot
        ):
            text = host.format_host_snapshot(app, compact=True)
        self.assertIn("a[b]c[/x]\\x0a", render(text).plain)


class BoundaryTests(_Base):
    def test_ac12_workspace_stays_install_root_and_launch_edits_ask(self) -> None:
        from harness.gate import load_harness

        install = self.make_dir("install")
        repo = self.make_dir("repo")
        launch.record_launch_dir(launch.encode_token(repo))
        harness = load_harness(install)
        self.assertEqual(harness.workspace_path(), install)
        target = repo / "f.py"
        target.write_text("x = 1\n", encoding="utf-8")
        asked = {"n": 0}

        def ask(_request) -> str:
            asked["n"] += 1
            return "deny"

        harness.authorize(
            "editor_propose_edit",
            {"path": str(target)},
            cwd=repo,
            target=target,
            ask=ask,
        )
        self.assertEqual(asked["n"], 1)

    def test_ac13_no_split_process_global(self) -> None:
        code = (
            "import sys; import sophon.cli.terminal as t; "
            "t.parse_chat_argv(['--no-spawn-window']); "
            "assert 'sophon.cli.host.launch' not in sys.modules, 'split'; "
            "assert 'cli.host.launch' in sys.modules; print('ok')"
        )
        env = {**os.environ, "SOPHON_SKIP_DOTENV": "1"}
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
