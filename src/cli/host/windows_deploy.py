from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from utils.device.env_bootstrap import sophon_project_root
from utils.device.platform import (
    is_windows_pe_executable,
    is_wsl,
    sophon_windows_root,
    sophon_windows_root_linux,
    store_pwsh_windows,
    windows_path_to_linux,
)

RSYNC_EXCLUDES = (
    ".venv/",
    ".venv-wsl/",
    ".git/",
    ".env",
    "__pycache__/",
    "*.egg-info/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "dist/",
    "build/",
    "models/",
    "data/chat_logs/",
    "data/chat_audio/",
    "data/memory/",
    "evaluation_runs/",
)


def windows_runtime_root() -> str:
    if sys.platform == "win32" and not os_env_windows_root_set():
        return str(sophon_project_root())
    return sophon_windows_root()


def os_env_windows_root_set() -> bool:
    import os

    return bool(os.environ.get("SOPHON_WINDOWS_ROOT", "").strip())


def linux_runtime_root() -> Path:
    if sys.platform == "win32":
        return Path(windows_runtime_root())
    linux = sophon_windows_root_linux()
    if linux is not None:
        return linux
    converted = windows_path_to_linux(windows_runtime_root())
    if converted:
        return Path(converted)
    raise RuntimeError(f"Cannot map Windows deploy root {windows_runtime_root()!r} into WSL.")


def windows_chat_tui_exe() -> Path | None:
    dest = linux_runtime_root() / ".venv" / "Scripts" / "sophon-chat-tui.exe"
    return dest if is_windows_pe_executable(dest) else None


def rsync_source_to_windows(source: Path, dest: Path) -> None:
    rsync = shutil.which("rsync")
    if rsync is None:
        raise RuntimeError("rsync is required to copy the WSL checkout onto the Windows deploy tree.")
    dest.mkdir(parents=True, exist_ok=True)
    args = [rsync, "-a", "--delete"]
    for pattern in RSYNC_EXCLUDES:
        args.extend(["--exclude", pattern])
    args.extend([f"{source}/", f"{dest}/"])
    proc = subprocess.run(args, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"rsync failed with exit code {proc.returncode}")


def run_store_pwsh(command: str, *, windows_cwd: str) -> int:
    cmd = shutil.which("cmd.exe")
    if cmd is None:
        raise RuntimeError("cmd.exe is not on PATH. Cannot invoke Store PowerShell from WSL.")
    line = subprocess.list2cmdline(
        [
            "pwsh.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]
    )
    return subprocess.call([cmd, "/c", line], cwd="/mnt/c/Windows")


def sync_windows_venv(*, extras: tuple[str, ...] = ("tui", "finetune")) -> int:
    # TODO: add tts/sst extras when the Windows TUI session needs those backends
    extra_flags = " ".join(f"--extra {name}" for name in extras)
    root = windows_runtime_root().replace("'", "''")
    command = (
        f"Set-Location -LiteralPath '{root}'; "
        "[System.Environment]::SetEnvironmentVariable('UV_LINK_MODE','copy','Process'); "
        f"uv sync {extra_flags}"
    )
    return run_store_pwsh(command, windows_cwd=windows_runtime_root())


def deploy_windows(
    *,
    sync_files: bool = True,
    sync_venv: bool | None = None,
    install_profile: bool = True,
) -> list[str]:
    from cli.host.profiles import install_windows_terminal_profile

    messages: list[str] = []
    source = sophon_project_root()
    dest = linux_runtime_root()
    win_root = windows_runtime_root()
    messages.append(f"WSL source: {source}")
    messages.append(f"Windows deploy root: {win_root}")
    messages.append(f"WSL dest: {dest}")

    if sync_files:
        if not is_wsl() and sys.platform != "win32":
            raise RuntimeError("File deploy is implemented for WSL -> Windows.")
        if is_wsl() and source.resolve() != dest.resolve():
            rsync_source_to_windows(source, dest)
            messages.append("Copied checkout onto the Windows deploy tree (rsync, .venv/models/.env kept).")
            from utils.device.env_bootstrap import merge_dotenv_keys

            merged = merge_dotenv_keys(source / ".env", dest / ".env")
            if merged:
                messages.append(
                    f"Merged {len(merged)} API key(s) into Windows .env ({', '.join(merged)}). Restart Nexus."
                )
            else:
                messages.append("Windows .env already has the WSL managed API keys (or WSL keys are empty).")
        else:
            messages.append("Skipped file copy (source and deploy root are the same).")

    if install_profile:
        path = install_windows_terminal_profile()
        if path is not None:
            messages.append(f"Windows Terminal profile written: {path}")
            messages.append(
                f"Profile commandline uses Store PowerShell ({store_pwsh_windows()}) "
                "and scripts/start-sophon-chat.ps1 -Foreground."
            )

    exe = windows_chat_tui_exe()
    need_venv = exe is None if sync_venv is None else sync_venv
    if need_venv:
        messages.append("Running uv sync on Windows via Store pwsh.exe (this can take a while).")
        code = sync_windows_venv()
        if code != 0:
            messages.append(f"uv sync exited {code}. Re-run: python3 -m cli.host.windows_deploy --sync-venv")
        else:
            messages.append("Windows .venv sync finished.")
    elif exe is not None:
        messages.append(f"Windows TUI exe present: {exe}")

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the WSL sophon checkout to Windows and update the Terminal profile."
    )
    parser.add_argument("--sync-venv", action="store_true", default=False)
    parser.add_argument("--skip-venv", action="store_true", default=False)
    parser.add_argument("--profile-only", action="store_true", default=False)
    args = parser.parse_args()
    if args.profile_only or args.skip_venv:
        sync_venv: bool | None = False
    elif args.sync_venv:
        sync_venv = True
    else:
        sync_venv = None
    try:
        lines = deploy_windows(
            sync_files=not args.profile_only,
            sync_venv=sync_venv,
            install_profile=True,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    for line in lines:
        print(line, flush=True)


if __name__ == "__main__":
    main()
