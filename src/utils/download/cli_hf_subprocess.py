from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from backend.hf.registry import HF_MODEL_PRESETS, default_preset_key, resolve_preset_dir
from utils.device.env_bootstrap import load_mithril_dotenv, mithril_project_root


def build_hf_download_argv(
    repo_id: str,
    local_dir: Path,
    *,
    revision: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "huggingface_hub.cli.hf",
        "download",
        repo_id,
        "--local-dir",
        str(local_dir),
    ]
    if revision:
        cmd.extend(["--revision", revision])
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        cmd.extend(["--token", tok])
    return cmd


def _mask_argv_for_print(cmd: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "--token" and i + 1 < len(cmd):
            out.extend(["--token", "***"])
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    return out


def format_hf_download_cmd_for_display(cmd: list[str]) -> str:
    return " ".join(_mask_argv_for_print(cmd))


def main() -> None:
    load_mithril_dotenv()
    preset_keys = sorted(HF_MODEL_PRESETS.keys())
    parser = argparse.ArgumentParser(
        description=(
            "Download a Hub model using the official Hugging Face CLI "
            "(python -m huggingface_hub.cli.hf download). "
            "Loads .env / HF_TOKEN like other mithril commands."
        ),
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=None,
        help="Hub repo id. Omit with --preset, or omit both to use MITHRIL_HF_PRESET/model scan/registry fallback.",
    )
    parser.add_argument(
        "--preset",
        choices=preset_keys,
        default=None,
        help="Preset from mithril HF registry (backend.hf.registry). Omit repo_id alone for models/<slug>; omit both for default_preset_key().",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Destination. Default: preset layout or models/<repo-id-with-dashes>.",
    )
    parser.add_argument("--revision", default=None, help="Optional branch, tag, or commit.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Base for relative paths (default: directory containing pyproject.toml).",
    )
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve() if args.project_root else mithril_project_root()

    if args.preset is not None:
        effective_preset: str | None = args.preset
    elif args.repo_id is None:
        effective_preset = default_preset_key()
    else:
        effective_preset = None

    revision = args.revision
    if revision is None:
        revision = os.environ.get("MITHRIL_HF_REVISION", "").strip() or None

    repo_id: str
    dest: Path

    if effective_preset is not None:
        preset = HF_MODEL_PRESETS[effective_preset]
        repo_id = args.repo_id if args.repo_id else preset.repo_id
        if effective_preset == "gemma4_31b_it" and revision is None:
            revision = os.environ.get("GEMMA4_REVISION", "").strip() or None
        if args.local_dir:
            dest = Path(args.local_dir).expanduser()
            dest = dest if dest.is_absolute() else (root / dest).resolve()
        else:
            dest = resolve_preset_dir(effective_preset, root)
    else:
        assert args.repo_id is not None
        repo_id = args.repo_id
        if args.local_dir:
            dest = Path(args.local_dir).expanduser()
            dest = dest if dest.is_absolute() else (root / dest).resolve()
        else:
            dest = (root / "models" / repo_id.replace("/", "-")).resolve()

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_hf_download_argv(repo_id, dest, revision=revision)
    print("mithril-hf-download:", " ".join(_mask_argv_for_print(cmd)), file=sys.stderr, flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(root)))


if __name__ == "__main__":
    main()
