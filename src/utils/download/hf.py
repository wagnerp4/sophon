from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars
from tqdm.auto import tqdm as TqdmAuto

from backend.hf.registry import HF_MODEL_PRESETS, resolve_preset_dir


def hub_token_for_snapshot() -> str | bool:
    raw = os.environ.get("HF_TOKEN", "").strip()
    return raw if raw else True


def configure_hub_verbose(*, force_tqdm: bool = True) -> None:
    import logging

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
            self._mithril_emit()

        def _mithril_total_n(self) -> tuple[int, int]:
            tot_raw = getattr(self, "total", None)
            if tot_raw is None or (isinstance(tot_raw, float) and math.isnan(tot_raw)):
                total = 0
            else:
                total = max(0, int(tot_raw))
            n_raw = getattr(self, "n", 0)
            n = max(0, int(n_raw))
            return n, total

        def _mithril_emit(self) -> None:
            n, total = self._mithril_total_n()
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
            self._mithril_emit()
            return r

        def refresh(self, nolock: bool = False, lock_args=None) -> None:
            super().refresh(nolock=nolock, lock_args=lock_args)
            self._mithril_emit()

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
            super().set_description(desc, refresh=refresh)
            self._mithril_emit()

        def close(self) -> None:
            try:
                self._mithril_emit()
            finally:
                super().close()

    return _HubTqdmBridge


def snapshot_hf_files(
    repo_id: str,
    local_dir: Path,
    revision: str | None,
    tqdm_class: type | None,
    *,
    verbose: bool,
    preset_key_for_log: str | None = None,
) -> Path:
    resolved = local_dir.expanduser().resolve()
    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "local_dir": str(resolved),
        "token": hub_token_for_snapshot(),
    }
    if revision:
        kwargs["revision"] = revision
    if tqdm_class is not None:
        kwargs["tqdm_class"] = tqdm_class
    if verbose:
        if preset_key_for_log is not None:
            print(
                f"mithril: Hub pull preset={preset_key_for_log!r} repo_id={repo_id!r}",
                file=sys.stderr,
                flush=True,
            )
            print(f"mithril: local_dir={resolved}", file=sys.stderr, flush=True)
        else:
            print(
                f"mithril: Hub pull repo_id={repo_id!r} local_dir={resolved}",
                file=sys.stderr,
                flush=True,
            )
        print(
            "mithril: calling snapshot_download (repo metadata and file list can take minutes on first contact)…",
            file=sys.stderr,
            flush=True,
        )
    snapshot_download(**kwargs)
    if verbose:
        print("mithril: snapshot_download finished.", file=sys.stderr, flush=True)
    return resolved


def download_preset_snapshot(
    preset_key: str,
    tqdm_class: type | None = None,
    *,
    verbose: bool = False,
) -> Path:
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
    return snapshot_hf_files(
        repo_id,
        local_path,
        revision,
        tqdm_class,
        verbose=verbose,
        preset_key_for_log=preset_key,
    )


@dataclass(frozen=True)
class HFHubResolvedPull:
    repo_id: str
    local_path: Path
    revision: str | None


def resolve_hf_hub_pull_paths(
    preset_key: str,
    repo_id: str | None,
    local_dir: Path | None,
    revision: str | None,
    *,
    cwd: Path | None = None,
) -> HFHubResolvedPull:
    preset = HF_MODEL_PRESETS[preset_key]
    base = cwd if cwd is not None else Path.cwd()
    resolved_repo_id = repo_id or preset.repo_id
    if preset_key == "gemma4_31b_it" and repo_id is None:
        resolved_repo_id = os.environ.get("GEMMA4_REPO_ID", resolved_repo_id)
    resolved_revision = revision
    if resolved_revision is None and preset_key == "gemma4_31b_it":
        gr = os.environ.get("GEMMA4_REVISION")
        resolved_revision = gr if gr else None
    if local_dir is not None:
        lp = Path(local_dir).expanduser()
        lp = lp if lp.is_absolute() else (base / lp).resolve()
    else:
        default_local = preset.default_local_dir
        if repo_id is not None:
            default_local = str(Path("models") / resolved_repo_id.replace("/", "-"))
        if preset_key == "gemma4_31b_it" and repo_id is None:
            default_local = os.environ.get("GEMMA4_LOCAL_DIR", default_local)
        lp_rel = Path(default_local).expanduser()
        lp = lp_rel if lp_rel.is_absolute() else (base / lp_rel).resolve()
    return HFHubResolvedPull(repo_id=resolved_repo_id, local_path=lp, revision=resolved_revision)


def download_hf(
    *,
    preset: str,
    repo_id: str | None = None,
    local_dir: Path | str | None = None,
    revision: str | None = None,
    cwd: Path | None = None,
    tqdm_class: type | None = None,
    verbose: bool = False,
) -> Path:
    if preset not in HF_MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset!r}")
    spec = resolve_hf_hub_pull_paths(preset, repo_id, local_dir, revision, cwd=cwd)
    return snapshot_hf_files(
        spec.repo_id,
        spec.local_path,
        spec.revision,
        tqdm_class,
        verbose=verbose,
        preset_key_for_log=preset,
    )


def download_hf_by_repo_only(
    repo_id: str,
    *,
    local_dir: Path | str | None = None,
    revision: str | None = None,
    cwd: Path | None = None,
    tqdm_class: type | None = None,
    verbose: bool = False,
) -> Path:
    base = cwd if cwd is not None else Path.cwd()
    if local_dir is not None:
        lp = Path(local_dir).expanduser()
        lp = lp if lp.is_absolute() else (base / lp).resolve()
    else:
        lp = (base / "models" / repo_id.replace("/", "-")).resolve()
    return snapshot_hf_files(repo_id, lp, revision, tqdm_class, verbose=verbose)

