from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
from huggingface_hub.utils import enable_progress_bars
from tqdm.auto import tqdm as TqdmAuto

from utils.env_bootstrap import DOTENV_LOAD_PATH, load_lmwrap_dotenv
from utils.registry import HF_MODEL_PRESETS, preset_summary_lines, resolve_preset_dir

_DEBUG_DEFAULT_PRESET = "llama2_7b_chat"


def _hub_token_for_snapshot() -> str | bool:
    raw = os.environ.get("HF_TOKEN", "").strip()
    return raw if raw else True


def _configure_hub_verbose(*, force_tqdm: bool = True) -> None:
    import logging
    import sys

    enable_progress_bars()
    if force_tqdm:
        os.environ.setdefault("TQDM_POSITION", "-1")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for name in ("huggingface_hub", "httpx"):
        logging.getLogger(name).setLevel(logging.INFO)


def hub_tqdm_bridge_factory(
    on_progress: Callable[[int, int, str], None],
) -> type:
    class _HubTqdmBridge(TqdmAuto):
        def __init__(self, *args, **kwargs) -> None:
            kwargs.pop("name", None)
            kwargs["disable"] = True
            kwargs.setdefault("mininterval", 0.2)
            super().__init__(*args, **kwargs)
            self._lmwrap_emit()

        def _lmwrap_total_n(self) -> tuple[int, int]:
            tot_raw = getattr(self, "total", None)
            if tot_raw is None or (isinstance(tot_raw, float) and math.isnan(tot_raw)):
                total = 0
            else:
                total = max(0, int(tot_raw))
            n_raw = getattr(self, "n", 0)
            n = max(0, int(n_raw))
            return n, total

        def _lmwrap_emit(self) -> None:
            n, total = self._lmwrap_total_n()
            desc = str(getattr(self, "desc", "") or "").strip()
            unit = str(getattr(self, "unit", "") or "")
            unit_scale = bool(getattr(self, "unit_scale", False))

            if unit == "B" and unit_scale:

                def fmt(x: float) -> str:
                    ax = abs(x)
                    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
                        if ax >= div:
                            return f"{x / div:.2f} {label}"
                    return f"{x:.0f} B"

                if total > 0:
                    detail = f"{fmt(float(n))} / {fmt(float(total))}"
                else:
                    detail = fmt(float(n))
            elif total > 0:
                detail = f"{n} / {total}" + (f" {unit}" if unit else "")
            else:
                detail = f"{n}" + (f" {unit}" if unit else "")

            if desc:
                label = f"{desc} · {detail}"
            else:
                label = detail
            on_progress(n, total, label)

        def update(self, n: int | float | None = 1) -> bool | None:
            r = super().update(n)
            self._lmwrap_emit()
            return r

        def refresh(self, nolock: bool = False, lock_args=None) -> None:
            super().refresh(nolock=nolock, lock_args=lock_args)
            self._lmwrap_emit()

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
            super().set_description(desc, refresh=refresh)
            self._lmwrap_emit()

        def close(self) -> None:
            try:
                self._lmwrap_emit()
            finally:
                super().close()

    return _HubTqdmBridge


def download_preset_snapshot(
    preset_key: str,
    tqdm_class: type | None = None,
    *,
    verbose: bool = False,
) -> Path:
    import sys

    if preset_key not in HF_MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset_key!r}")
    preset = HF_MODEL_PRESETS[preset_key]
    repo_id = preset.repo_id
    if preset_key == "gemma4_31b_it":
        repo_id = os.environ.get("GEMMA4_REPO_ID", repo_id)
    local_path = resolve_preset_dir(preset_key)
    revision: str | None = None
    if preset_key == "gemma4_31b_it":
        raw = os.environ.get("GEMMA4_REVISION", "").strip()
        revision = raw or None
    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "local_dir": str(local_path),
        "token": _hub_token_for_snapshot(),
    }
    if revision:
        kwargs["revision"] = revision
    if tqdm_class is not None:
        kwargs["tqdm_class"] = tqdm_class
    if verbose:
        print(
            f"lmwrap: Hub pull preset={preset_key!r} repo_id={repo_id!r}",
            file=sys.stderr,
            flush=True,
        )
        print(f"lmwrap: local_dir={local_path}", file=sys.stderr, flush=True)
        print(
            "lmwrap: calling snapshot_download (repo metadata and file list can take minutes on first contact)…",
            file=sys.stderr,
            flush=True,
        )
    snapshot_download(**kwargs)
    if verbose:
        print("lmwrap: snapshot_download finished.", file=sys.stderr, flush=True)
    return local_path


def main() -> None:
    load_lmwrap_dotenv()
    preset_keys = sorted(HF_MODEL_PRESETS.keys())
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model snapshots via preset name or explicit repo id.",
    )
    parser.add_argument(
        "--preset",
        choices=preset_keys,
        default="llama2_7b_chat",
        help="Named Hub target from lmwrap.utils.registry.HF_MODEL_PRESETS (default: %(default)s).",
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
        help="Log to stderr, enable Hub progress bars, and print lmwrap status lines.",
    )
    args = parser.parse_args()

    if args.list_presets:
        print("\n".join(preset_summary_lines()))
        return

    if args.verbose:
        _configure_hub_verbose()

    preset = HF_MODEL_PRESETS[args.preset]
    repo_id = args.repo_id or preset.repo_id

    if args.preset == "gemma4_31b_it" and args.repo_id is None:
        repo_id = os.environ.get("GEMMA4_REPO_ID", repo_id)

    if args.local_dir is not None:
        local_path = args.local_dir
    else:
        default_local = preset.default_local_dir
        if args.repo_id is not None:
            default_local = str(Path("models") / args.repo_id.replace("/", "-"))
        if args.preset == "gemma4_31b_it" and args.repo_id is None:
            default_local = os.environ.get("GEMMA4_LOCAL_DIR", default_local)
        local_path = Path(default_local)

    revision = args.revision
    if revision is None and args.preset == "gemma4_31b_it":
        revision = os.environ.get("GEMMA4_REVISION")

    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "local_dir": str(local_path.expanduser().resolve()),
        "token": _hub_token_for_snapshot(),
    }
    if revision:
        kwargs["revision"] = revision
    if args.verbose:
        print(
            f"lmwrap: Hub pull preset={args.preset!r} repo_id={repo_id!r} local_dir={local_path!s}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "lmwrap: snapshot_download starting (metadata phase may stall with no tqdm yet)…",
            file=sys.stderr,
            flush=True,
        )
    out = snapshot_download(**kwargs)
    if args.verbose:
        print("lmwrap: snapshot_download finished.", file=sys.stderr, flush=True)
    print(out)


def main_debug() -> None:
    load_lmwrap_dotenv()
    preset_keys = sorted(HF_MODEL_PRESETS.keys())
    default_key = os.environ.get("LMWRAP_DEBUG_PRESET", _DEBUG_DEFAULT_PRESET).strip()
    if default_key not in HF_MODEL_PRESETS:
        default_key = _DEBUG_DEFAULT_PRESET
    parser = argparse.ArgumentParser(
        description=(
            "Verbose terminal-only Hub pull for debugging: tqdm + INFO logs on stderr. "
            "Default preset is llama2_7b_chat (same local_dir layout as the registry preset). "
            "Override with --preset or LMWRAP_DEBUG_PRESET."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=preset_keys,
        default=default_key,
        help="Registry preset (default: %(default)s or LMWRAP_DEBUG_PRESET env).",
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

    _configure_hub_verbose()
    preset = HF_MODEL_PRESETS[args.preset]
    print(f"lmwrap-debug: cwd={Path.cwd()}", file=sys.stderr, flush=True)
    tok = os.environ.get("HF_TOKEN", "").strip()
    print(
        f"lmwrap-debug: dotenv={DOTENV_LOAD_PATH!s} HF_TOKEN={'set' if tok else 'unset'}",
        file=sys.stderr,
        flush=True,
    )
    print(f"lmwrap-debug: preset={args.preset!r} repo_id={preset.repo_id!r}", file=sys.stderr, flush=True)
    print(f"lmwrap-debug: local_dir={resolve_preset_dir(args.preset)}", file=sys.stderr, flush=True)
    if "llama" in args.preset.lower():
        print(
            "lmwrap-debug: Llama repos are gated. Set HF_TOKEN or run `huggingface-cli login`,",
            file=sys.stderr,
            flush=True,
        )
        print(
            "lmwrap-debug: and open the model card on the Hub to accept the license for your account.",
            file=sys.stderr,
            flush=True,
        )
    try:
        out = download_preset_snapshot(args.preset, tqdm_class=None, verbose=True)
    except GatedRepoError as e:
        gated = getattr(e, "repo_id", preset.repo_id)
        print(
            f"lmwrap-debug: gated repo {gated!r} — access denied for this token/account.",
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
