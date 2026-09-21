from __future__ import annotations

import logging
import math
import os
import re
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars
from tqdm.auto import tqdm as TqdmAuto

from backend.hf.registry import HF_MODEL_PRESETS, resolve_preset_dir


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".incomplete", ".part")
_FILE_COUNT_TOTAL_MAX = 4096
_SIZE_GIB_RE = re.compile(r"~(\d+(?:\.\d+)?)\s*GiB", re.IGNORECASE)


def scan_local_dir_download_bytes(local_dir: Path) -> int:
    total = 0
    if not local_dir.is_dir():
        return 0
    seen: set[str] = set()

    def add_file(path: Path) -> None:
        nonlocal total
        try:
            key = str(path.resolve())
        except OSError:
            return
        if key in seen or not path.is_file():
            return
        seen.add(key)
        try:
            total += path.stat().st_size
        except OSError:
            pass

    cache_root = local_dir / ".cache"
    if cache_root.is_dir():
        for path in cache_root.rglob("*"):
            if path.is_file():
                add_file(path)
    try:
        entries = list(local_dir.iterdir())
    except OSError:
        entries = []
    for path in entries:
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(_WEIGHT_SUFFIXES):
            add_file(path)
    for shard in local_dir.glob("model-*-of-*.safetensors"):
        add_file(shard)
    return total


def probe_hub_snapshot_bytes(repo_id: str) -> int:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception:
        return 0
    total = 0
    for sibling in getattr(info, "siblings", None) or []:
        size = getattr(sibling, "size", None)
        if size is None:
            continue
        try:
            total += int(size)
        except (TypeError, ValueError):
            continue
    return total


def parse_hub_size_hint(text: str) -> int:
    match = _SIZE_GIB_RE.search(text)
    if match is None:
        return 0
    return int(float(match.group(1)) * (2**30))


@contextmanager
def watch_local_download_bytes(
    local_dir: Path,
    on_disk_bytes: Callable[[int], None],
    *,
    interval_s: float = 0.5,
) -> Iterator[None]:
    stop = threading.Event()

    def poll() -> None:
        last = -1
        while not stop.is_set():
            nbytes = scan_local_dir_download_bytes(local_dir)
            if nbytes != last:
                last = nbytes
                on_disk_bytes(nbytes)
            if stop.wait(interval_s):
                break

    thread = threading.Thread(target=poll, name="sophon-download-watch", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval_s + 1.0)


def hub_token_for_snapshot() -> str | bool:
    raw = os.environ.get("HF_TOKEN", "").strip()
    return raw if raw else True


def format_byte_count(value: int | float) -> str:
    amount = float(value)
    ax = abs(amount)
    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if ax >= div:
            return f"{amount / div:.2f} {label}"
    return f"{amount:.0f} B"


def format_byte_progress_pair(n: int, total: int) -> str:
    if total > 0:
        return f"{format_byte_count(n)} / {format_byte_count(total)}"
    if n > 0:
        return format_byte_count(n)
    return "0 B"


def format_percent(n: int, total: int) -> str:
    if total <= 0:
        return "?%"
    pct = min(100.0, max(0.0, (float(n) / float(total)) * 100.0))
    return f"{pct:.1f}%"


def format_eta_s(seconds: float) -> str:
    total = int(max(0, seconds))
    if total < 60:
        return f"{total}s"
    mins, secs = divmod(total, 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, rem = divmod(mins, 60)
    return f"{hours}h {rem}m"


def format_progress_bar(n: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "░" * width
    filled = min(width, int((n / total) * width))
    return ("█" * filled) + ("░" * (width - filled))


def format_download_bar_line(
    n: int,
    total: int,
    *,
    bytes_per_s: float = 0.0,
    file_count: bool = False,
) -> str:
    bar = format_progress_bar(n, total)
    pct = format_percent(n, total)
    if file_count:
        if total > 0:
            return f"[{bar}] {pct} · {n} / {total} files"
        return f"[{bar}] {pct} · enumerating repo files on Hub"
    detail = format_byte_progress_pair(n, total)
    parts = [f"[{bar}] {pct} · {detail}"]
    if bytes_per_s > 0:
        parts.append(f"{format_byte_count(bytes_per_s)}/s")
        if total > n:
            parts.append(f"ETA {format_eta_s((total - n) / bytes_per_s)}")
    elif n <= 0:
        parts.append("no bytes on disk yet")
    else:
        parts.append("size unchanged")
    return " · ".join(parts)


def normalize_hub_phase(desc: str) -> str:
    phase = desc.strip()
    if not phase:
        return "preparing download"
    if phase == "Downloading (incomplete total...)":
        return "waiting for first bytes (Hub is resolving shard sizes / XET handshake)"
    if phase.startswith("Fetching "):
        return phase
    if phase == "Download complete":
        return "download complete"
    return phase


def is_file_count_progress(n: int, total: int, desc: str) -> bool:
    if desc.startswith("Fetching "):
        return True
    if " files" in desc.lower():
        return True
    return total > 0 and total <= _FILE_COUNT_TOTAL_MAX and n <= total and n <= _FILE_COUNT_TOTAL_MAX


def format_idle_download_hint() -> str:
    return format_download_bar_line(0, 0)


def split_hub_progress_label(label: str) -> tuple[str, str]:
    text = label.strip()
    if " · " in text:
        phase, detail = text.split(" · ", 1)
        return normalize_hub_phase(phase), detail.strip()
    return normalize_hub_phase(text), format_byte_progress_pair(0, 0)


def format_download_progress(
    n: int,
    total: int,
    desc: str = "",
    *,
    bytes_per_s: float = 0.0,
) -> str:
    phase = normalize_hub_phase(desc)
    file_count = is_file_count_progress(n, total, desc)
    detail = format_download_bar_line(
        n,
        total,
        bytes_per_s=0.0 if file_count else bytes_per_s,
        file_count=file_count,
    )
    if phase:
        return f"{phase}\n{detail}"
    return detail


def _format_hub_log_line(logger_name: str, message: str) -> str | None:
    text = message.strip()
    if not text:
        return None
    if logger_name.startswith("huggingface_hub"):
        if "Number of files in the repo is unreliable" in text:
            return "hub: large repo — listing all files from Hub (can take a while)"
        if text.startswith("Fetching ") or "snapshot" in text.lower():
            return f"hub: {text}"
        return None
    if logger_name == "httpx":
        if "huggingface.co" not in text or "HTTP Request:" not in text:
            return None
        if "xet-read-token" in text:
            return "hub: XET read token OK — negotiating ~22 GiB weight stream (can take 1–5 min before first byte)"
        if "HEAD" in text and "model.safetensors" in text:
            return "hub: HEAD model.safetensors — size confirmed, preparing download"
        if "HEAD" in text:
            head_match = re.search(r"/(?:resolve|blob)/[^/]+/[^/]+/(.+?)(?:\s|$)", text)
            if head_match is not None:
                name = head_match.group(1).strip()
                return f"hub: HEAD {name}"
        file_match = re.search(r"/(?:resolve|blob)/[^/]+/[^/]+/(.+?)(?:\s|$)", text)
        if file_match is not None:
            name = file_match.group(1).strip()
            if len(name) > 72:
                name = f"(…){name[-68:]}"
            return f"hub: GET {name}"
        repo_match = re.search(r"huggingface\.co/(?:api/)?models/([^?\s\"/]+/[^?\s\"/]+)", text)
        if repo_match is not None:
            return f"hub: resolving {repo_match.group(1)}"
    return None


@contextmanager
def capture_hub_download_logs(
    emit: Callable[[str], None],
    *,
    min_interval_s: float = 0.8,
) -> Iterator[None]:
    state = {"last_emit": 0.0, "seen": set()}

    class _HubLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            line = _format_hub_log_line(record.name, record.getMessage())
            if line is None or line in state["seen"]:
                return
            urgent = (
                "XET read token" in line
                or "HEAD model.safetensors" in line
                or "listing all files" in line
            )
            now = time.monotonic()
            if not urgent and (now - state["last_emit"]) < min_interval_s:
                return
            state["last_emit"] = now
            state["seen"].add(line)
            emit(line)

    handler = _HubLogHandler()
    handler.setLevel(logging.INFO)
    targets = [
        logging.getLogger("huggingface_hub"),
        logging.getLogger("httpx"),
    ]
    previous_levels = [(logger, logger.level) for logger in targets]
    for logger in targets:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        handler.close()
        for logger, level in previous_levels:
            logger.removeHandler(handler)
            logger.setLevel(level)


def throttled_progress_callback(
    on_progress: Callable[[int, int, str], None],
    *,
    min_interval_s: float = 0.2,
) -> Callable[[int, int, str], None]:
    state = {"last_emit": 0.0, "last_n": -1, "last_total": -1}

    def emit(n: int, total: int, label: str) -> None:
        now = time.monotonic()
        done = total > 0 and n >= total
        changed = n != state["last_n"] or total != state["last_total"]
        if not done and not changed and (now - state["last_emit"]) < min_interval_s:
            return
        state["last_emit"] = now
        state["last_n"] = n
        state["last_total"] = total
        on_progress(n, total, label)

    return emit


@contextmanager
def capture_hub_user_warnings(emit: Callable[[str], None]) -> Iterator[None]:
    seen: set[str] = set()

    def showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: object | None = None,
        line: str | None = None,
    ) -> None:
        text = str(message).strip()
        if not text or text in seen:
            return
        seen.add(text)
        emit(text)

    previous = warnings.showwarning
    warnings.showwarning = showwarning
    try:
        yield
    finally:
        warnings.showwarning = previous


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
            kwargs.setdefault("mininterval", 0.2)
            kwargs.setdefault("leave", False)
            kwargs.setdefault("disable", False)
            if kwargs.get("file") is None:
                kwargs["file"] = open(os.devnull, "w", encoding="utf-8")
            super().__init__(*args, **kwargs)
            self._sophon_emit()

        def _sophon_total_n(self) -> tuple[int, int]:
            tot_raw = getattr(self, "total", None)
            if tot_raw is None or (isinstance(tot_raw, float) and math.isnan(tot_raw)):
                total = 0
            else:
                total = max(0, int(tot_raw))
            n_raw = getattr(self, "n", 0)
            n = max(0, int(n_raw))
            return n, total

        def _sophon_emit(self) -> None:
            n, total = self._sophon_total_n()
            desc = str(getattr(self, "desc", "") or "").strip()
            unit = str(getattr(self, "unit", "") or "")
            unit_scale = bool(getattr(self, "unit_scale", False))
            use_bytes = (unit == "B" and unit_scale) or total >= 1024 or n >= 1024
            if use_bytes:
                label = format_download_progress(n, total, desc)
            elif is_file_count_progress(n, total, desc):
                if total > 0:
                    detail = f"{n} / {total} files"
                else:
                    detail = "enumerating files..."
                label = f"{desc} · {detail}" if desc else detail
            elif total > 0:
                detail = f"{n} / {total}" + (f" {unit}" if unit else "")
                label = f"{desc} · {detail}" if desc else detail
            else:
                detail = f"{n}" + (f" {unit}" if unit else "")
                label = f"{desc} · {detail}" if desc else detail
            on_progress(n, total, label)

        def update(self, n: int | float | None = 1) -> bool | None:
            r = super().update(n)
            self._sophon_emit()
            return r

        def refresh(self, nolock: bool = False, lock_args=None) -> None:
            super().refresh(nolock=nolock, lock_args=lock_args)
            self._sophon_emit()

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
            super().set_description(desc, refresh=refresh)
            self._sophon_emit()

        def close(self) -> None:
            try:
                self._sophon_emit()
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
    on_status: Callable[[str], None] | None = None,
    on_expected_bytes: Callable[[int], None] | None = None,
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
    if tqdm_class is None and verbose:
        configure_hub_verbose(force_tqdm=True)
    if on_status is not None:
        on_status(f"resolving {repo_id} on Hugging Face Hub")
        on_status(f"target directory: {resolved}")
        on_status("querying Hub for snapshot size")
    expected = probe_hub_snapshot_bytes(repo_id)
    if expected > 0:
        if on_status is not None:
            on_status(f"snapshot size {format_byte_count(expected)}")
        if on_expected_bytes is not None:
            on_expected_bytes(expected)
    elif on_status is not None:
        on_status("Hub did not return file sizes yet. Progress uses on-disk bytes until tqdm reports a total")
    if on_status is not None:
        on_status("fetching repo metadata and file list (first contact can take minutes)")
    if verbose:
        if preset_key_for_log is not None:
            print(
                f"sophon: Hub pull preset={preset_key_for_log!r} repo_id={repo_id!r}",
                file=sys.stderr,
                flush=True,
            )
            print(f"sophon: local_dir={resolved}", file=sys.stderr, flush=True)
        else:
            print(
                f"sophon: Hub pull repo_id={repo_id!r} local_dir={resolved}",
                file=sys.stderr,
                flush=True,
            )
        print(
            "sophon: calling snapshot_download (repo metadata and file list can take minutes on first contact)…",
            file=sys.stderr,
            flush=True,
        )
    snapshot_download(**kwargs)
    if on_status is not None:
        on_status("snapshot_download finished")
    if verbose:
        print("sophon: snapshot_download finished.", file=sys.stderr, flush=True)
    return resolved


def download_preset_snapshot(
    preset_key: str,
    tqdm_class: type | None = None,
    *,
    verbose: bool = False,
    on_status: Callable[[str], None] | None = None,
    on_expected_bytes: Callable[[int], None] | None = None,
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
        on_status=on_status,
        on_expected_bytes=on_expected_bytes,
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

