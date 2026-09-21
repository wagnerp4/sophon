from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub.errors import GatedRepoError

from backend.hf.registry import HF_MODEL_PRESETS, preset_summary_lines, resolve_preset_dir
from utils.device.env_bootstrap import DOTENV_LOAD_PATH, load_sophon_dotenv
from utils.download.hf import (
    configure_hub_verbose,
    download_preset_snapshot,
    resolve_hf_hub_pull_paths,
    snapshot_hf_files,
)

_DEBUG_DEFAULT_PRESET = "llama2_7b_chat"


def main() -> None:
    load_sophon_dotenv()
    preset_keys = sorted(HF_MODEL_PRESETS.keys())
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model snapshots via preset name or explicit repo id.",
    )
    parser.add_argument(
        "--preset",
        choices=preset_keys,
        default="llama2_7b_chat",
        help="Named Hub target from backend.hf.registry.HF_MODEL_PRESETS (default: %(default)s).",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Override preset Hub model id.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Directory for snapshot files (default from preset or derived from repo id).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision (branch name, tag, or commit).",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print preset keys and repo ids, then exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log to stderr, enable Hub progress bars, and print sophon status lines.",
    )
    args = parser.parse_args()

    if args.list_presets:
        print("\n".join(preset_summary_lines()))
        return

    if args.verbose:
        configure_hub_verbose()

    spec = resolve_hf_hub_pull_paths(
        args.preset,
        args.repo_id,
        args.local_dir,
        args.revision,
    )

    snapshot_hf_files(
        spec.repo_id,
        spec.local_path,
        spec.revision,
        None,
        verbose=args.verbose,
        preset_key_for_log=args.preset,
    )
    print(spec.local_path)


def main_debug() -> None:
    load_sophon_dotenv()
    preset_keys = sorted(HF_MODEL_PRESETS.keys())
    default_key = os.environ.get("SOPHON_DEBUG_PRESET", _DEBUG_DEFAULT_PRESET).strip()
    if default_key not in HF_MODEL_PRESETS:
        default_key = _DEBUG_DEFAULT_PRESET
    parser = argparse.ArgumentParser(
        description=(
            "Verbose terminal-only Hub pull for debugging: tqdm + INFO logs on stderr. "
            "Default preset is llama2_7b_chat (same local_dir layout as the registry preset). "
            "Override with --preset or SOPHON_DEBUG_PRESET."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=preset_keys,
        default=default_key,
        help="Registry preset (default: %(default)s or SOPHON_DEBUG_PRESET env).",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print preset keys and repo ids, then exit.",
    )
    args = parser.parse_args()

    if args.list_presets:
        print("\n".join(preset_summary_lines()))
        return

    configure_hub_verbose()
    preset = HF_MODEL_PRESETS[args.preset]
    tok = os.environ.get("HF_TOKEN", "").strip()
    print(f"sophon-debug: cwd={Path.cwd()}", file=sys.stderr, flush=True)
    print(
        f"sophon-debug: dotenv={DOTENV_LOAD_PATH!s} HF_TOKEN={'set' if tok else 'unset'}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"sophon-debug: preset={args.preset!r} repo_id={preset.repo_id!r}",
        file=sys.stderr,
        flush=True,
    )
    print(f"sophon-debug: local_dir={resolve_preset_dir(args.preset)}", file=sys.stderr, flush=True)
    if "llama" in args.preset.lower():
        print(
            "sophon-debug: Llama repos are gated. Set HF_TOKEN or run `huggingface-cli login`,",
            file=sys.stderr,
            flush=True,
        )
        print(
            "sophon-debug: and open the model card on the Hub to accept the license for your account.",
            file=sys.stderr,
            flush=True,
        )
    try:
        out = download_preset_snapshot(args.preset, tqdm_class=None, verbose=True)
    except GatedRepoError as e:
        gated = getattr(e, "repo_id", preset.repo_id)
        print(
            f"sophon-debug: gated repo {gated!r} — access denied for this token/account.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from e
    print(out)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "debug":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main_debug()
    else:
        main()
