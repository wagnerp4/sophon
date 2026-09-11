import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import snapshot_download

from backend.hf.registry import HF_MODEL_PRESETS, resolve_preset_dir
from utils.download.hf import (
    capture_hub_download_logs,
    configure_hub_verbose,
    download_preset_snapshot,
    hub_tqdm_bridge_factory,
    throttled_progress_callback,
    watch_local_download_bytes,
)


class _TimingState:
    def __init__(self) -> None:
        self.start = time.monotonic()
        self.first_progress_at: float | None = None
        self.first_byte_at: float | None = None
        self.last_n = 0
        self.last_total = 0
        self.last_label = ""
        self.events: list[str] = []

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def on_progress(self, n: int, total: int, label: str) -> None:
        now = self.elapsed()
        if self.first_progress_at is None:
            self.first_progress_at = now
            self.events.append(f"+{now:6.1f}s  first tqdm callback  n={n} total={total}  {label[:80]}")
        if n > 0 and self.first_byte_at is None:
            self.first_byte_at = now
            self.events.append(f"+{now:6.1f}s  first non-zero byte  n={n} total={total}")
        if n != self.last_n or total != self.last_total:
            self.last_n = n
            self.last_total = total
            self.last_label = label
            if n > 0 and (n % (50 * 1024 * 1024) < 1024 * 1024 or n == total):
                line = (
                    f"+{now:6.1f}s  tqdm  {n / (1024**3):.2f} GiB / "
                    f"{(total / (1024**3) if total else 0):.2f} GiB"
                )
                self.events.append(line)
                print(line, flush=True)

    def on_log(self, text: str) -> None:
        line = f"+{self.elapsed():6.1f}s  log  {text[:120]}"
        self.events.append(line)
        print(line, flush=True)

    def on_disk(self, nbytes: int) -> None:
        now = self.elapsed()
        line = f"+{now:6.1f}s  disk  {nbytes / (1024**3):.2f} GiB in .cache (tqdm may stay at 0 during XET)"
        if not self.events or self.events[-1] != line:
            self.events.append(line)
            print(line, flush=True)


def run_direct(repo_id: str, local_dir: Path, token: str | bool) -> _TimingState:
    state = _TimingState()
    tqdm_class = hub_tqdm_bridge_factory(throttled_progress_callback(state.on_progress, min_interval_s=0.5))
    print(f"[direct] repo_id={repo_id}", flush=True)
    print(f"[direct] local_dir={local_dir}", flush=True)
    print("[direct] snapshot_download (native HF path, tqdm bridged for timing only)", flush=True)
    with watch_local_download_bytes(local_dir, state.on_disk):
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            tqdm_class=tqdm_class,
        )
    return state


def run_orodruin(preset_key: str) -> _TimingState:
    state = _TimingState()

    def on_status(text: str) -> None:
        state.on_log(f"status: {text}")

    tqdm_class = hub_tqdm_bridge_factory(throttled_progress_callback(state.on_progress, min_interval_s=0.5))
    configure_hub_verbose(force_tqdm=True)
    os.environ["TQDM_POSITION"] = "-1"
    print(f"[orodruin] preset={preset_key}", flush=True)
    print("[orodruin] download_preset_snapshot + verbose logging + log capture (same as TUI worker)", flush=True)
    local_dir = resolve_preset_dir(preset_key, ROOT)
    with watch_local_download_bytes(local_dir, state.on_disk):
        with capture_hub_download_logs(state.on_log, min_interval_s=0.8):
            download_preset_snapshot(
                preset_key,
                tqdm_class=tqdm_class,
                verbose=True,
                on_status=on_status,
            )
    return state


def _safe_console(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def print_report(mode: str, state: _TimingState) -> None:
    print(f"\n=== {mode} report ===", flush=True)
    print(f"total elapsed: {state.elapsed():.1f}s", flush=True)
    if state.first_progress_at is not None:
        print(f"first tqdm callback: {state.first_progress_at:.1f}s", flush=True)
    else:
        print("first tqdm callback: never", flush=True)
    if state.first_byte_at is not None:
        print(f"first non-zero byte: {state.first_byte_at:.1f}s", flush=True)
    else:
        print("first non-zero byte: never", flush=True)
    print(f"final n={state.last_n} total={state.last_total}", flush=True)
    for line in state.events:
        print(_safe_console(line), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct vs orodruin Hub download wiring.")
    parser.add_argument("--mode", choices=("direct", "orodruin", "both"), default="direct")
    parser.add_argument("--preset", default="gemma4_12b_it")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Override destination (default: registry preset dir). Use a separate dir for side-by-side tests.",
    )
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN", "").strip() or True
    preset_key = args.preset
    local_dir = (
        Path(args.local_dir).expanduser().resolve()
        if args.local_dir
        else resolve_preset_dir(preset_key, ROOT)
    )
    repo_id = HF_MODEL_PRESETS[preset_key].repo_id

    if args.mode in ("direct", "both"):
        state = run_direct(repo_id, local_dir, token)
        print_report("direct", state)

    if args.mode in ("orodruin", "both"):
        if args.mode == "both" and not args.local_dir:
            print("\n[orodruin] skipped in 'both' without --local-dir (avoid writing same tree twice)", flush=True)
        else:
            state = run_orodruin(preset_key)
            print_report("orodruin", state)


if __name__ == "__main__":
    main()
