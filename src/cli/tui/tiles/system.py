from __future__ import annotations

import shutil
import subprocess
import time
from collections import deque
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, ProgressBar, Static

from cli.tui.tiles.base import BaseTile, TileState

_HISTORY_LEN = 48
_SPARK = "▁▂▃▄▅▆▇█"
_DOCKER_CACHE_S = 12.0

_net_prev: tuple[float, int, int] | None = None
_disk_io_prev: tuple[float, int, int] | None = None
_docker_cache: tuple[float, dict[str, Any]] | None = None


def _fmt_bytes(value: int | float) -> str:
    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= div:
            return f"{value / div:.2f} {label}"
    return f"{int(value)} B"


def _fmt_pair(used: int | float, total: int | float) -> str:
    return f"{_fmt_bytes(used)} / {_fmt_bytes(total)}"


def _fmt_rate(bps: float) -> str:
    if bps < 0:
        bps = 0.0
    if bps >= 2**20:
        return f"{bps / (2**20):.1f} MiB/s"
    if bps >= 2**10:
        return f"{bps / (2**10):.0f} KiB/s"
    return f"{bps:.0f} B/s"


def sparkline(values: deque[float] | list[float], *, width: int = 28) -> str:
    if not values:
        return "·" * max(width, 1)
    seq = list(values)
    if len(seq) > width:
        seq = seq[-width:]
    lo = min(seq)
    hi = max(seq)
    span = hi - lo
    if span <= 1e-9:
        mid = _SPARK[len(_SPARK) // 2]
        return mid * len(seq)
    out: list[str] = []
    last = len(_SPARK) - 1
    for value in seq:
        idx = int(round((value - lo) / span * last))
        out.append(_SPARK[max(0, min(last, idx))])
    return "".join(out)


def _collect_gpu_snapshot() -> dict[str, Any]:
    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError("pynvml unavailable (install nvidia-ml-py)") from exc
    pynvml.nvmlInit()
    try:
        count = int(pynvml.nvmlDeviceGetCount())
        if count <= 0:
            raise RuntimeError("no NVIDIA devices reported by NVML")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        try:
            temp = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            temp = None
        try:
            power_mw = int(pynvml.nvmlDeviceGetPowerUsage(handle))
            power_w = power_mw / 1000.0
        except Exception:
            power_w = None
        processes: list[dict[str, Any]] = []
        try:
            running = list(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
        except Exception:
            running = []
        try:
            import psutil
        except ImportError:
            psutil = None
        for proc in running:
            pid = int(getattr(proc, "pid", 0) or 0)
            used = int(getattr(proc, "usedGpuMemory", 0) or 0)
            proc_name = "?"
            if psutil is not None and pid > 0:
                try:
                    proc_name = psutil.Process(pid).name()
                except Exception:
                    proc_name = f"pid:{pid}"
            processes.append({"pid": pid, "name": proc_name, "vram_bytes": used})
        processes.sort(key=lambda row: int(row["vram_bytes"]), reverse=True)
        return {
            "name": str(name),
            "total_bytes": int(mem.total),
            "used_bytes": int(mem.used),
            "free_bytes": int(mem.free),
            "gpu_util": int(util.gpu),
            "mem_util": int(util.memory),
            "temp_c": temp,
            "power_w": power_w,
            "processes": processes,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _collect_host_resources() -> dict[str, Any]:
    import psutil

    cpu_util = float(psutil.cpu_percent(interval=None))
    freq = None
    try:
        freq_info = psutil.cpu_freq()
        if freq_info is not None:
            freq = float(freq_info.current)
    except Exception:
        freq = None
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "cpu": {
            "util": cpu_util,
            "count": int(psutil.cpu_count(logical=True) or 0),
            "freq_mhz": freq,
        },
        "ram": {
            "total_bytes": int(vm.total),
            "used_bytes": int(vm.used),
            "avail_bytes": int(vm.available),
            "percent": float(vm.percent),
        },
        "swap": {
            "total_bytes": int(swap.total),
            "used_bytes": int(swap.used),
            "percent": float(swap.percent),
        },
    }


def _collect_disk_io() -> dict[str, Any]:
    global _disk_io_prev
    import psutil

    now = time.time()
    counters = psutil.disk_io_counters()
    if counters is None:
        return {"read_bps": 0.0, "write_bps": 0.0, "ok": False, "error": "no disk io counters"}
    read_b = int(counters.read_bytes)
    write_b = int(counters.write_bytes)
    read_bps = 0.0
    write_bps = 0.0
    if _disk_io_prev is not None:
        prev_t, prev_r, prev_w = _disk_io_prev
        dt = max(1e-3, now - prev_t)
        read_bps = max(0.0, (read_b - prev_r) / dt)
        write_bps = max(0.0, (write_b - prev_w) / dt)
    _disk_io_prev = (now, read_b, write_b)
    return {
        "ok": True,
        "read_bps": read_bps,
        "write_bps": write_bps,
        "read_bytes": read_b,
        "write_bytes": write_b,
        "error": "",
    }


def _collect_network() -> dict[str, Any]:
    global _net_prev
    import psutil

    now = time.time()
    counters = psutil.net_io_counters()
    sent = int(counters.bytes_sent)
    recv = int(counters.bytes_recv)
    up_bps = 0.0
    down_bps = 0.0
    if _net_prev is not None:
        prev_t, prev_sent, prev_recv = _net_prev
        dt = max(1e-3, now - prev_t)
        up_bps = max(0.0, (sent - prev_sent) / dt)
        down_bps = max(0.0, (recv - prev_recv) / dt)
    _net_prev = (now, sent, recv)

    up_ifaces: list[str] = []
    try:
        stats_map = psutil.net_if_stats()
        for name, addrs in psutil.net_if_addrs().items():
            _ = addrs
            stats = stats_map.get(name)
            if stats is None or not stats.isup:
                continue
            if name.lower().startswith(("loopback", "lo")):
                continue
            up_ifaces.append(name)
    except Exception:
        up_ifaces = []

    return {
        "up_bps": up_bps,
        "down_bps": down_bps,
        "bytes_sent": sent,
        "bytes_recv": recv,
        "ifaces": up_ifaces[:4],
    }


def _docker_cmd(args: list[str], *, timeout_s: float = 2.5) -> str:
    exe = shutil.which("docker")
    if not exe:
        raise RuntimeError("docker not on PATH")
    completed = subprocess.run(
        [exe, *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "docker failed").strip()
        raise RuntimeError(err.splitlines()[0][:160] if err else "docker failed")
    return completed.stdout or ""


def _collect_docker() -> dict[str, Any]:
    global _docker_cache
    now = time.time()
    if _docker_cache is not None and (now - _docker_cache[0]) < _DOCKER_CACHE_S:
        return dict(_docker_cache[1])
    try:
        running = [line for line in _docker_cmd(["ps", "-q"]).splitlines() if line.strip()]
        images = [line for line in _docker_cmd(["images", "-q"]).splitlines() if line.strip()]
        names = [
            line.strip()
            for line in _docker_cmd(["ps", "--format", "{{.Names}}"]).splitlines()
            if line.strip()
        ]
        payload = {
            "ok": True,
            "containers": len(running),
            "images": len(set(images)),
            "names": names[:6],
            "error": "",
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "containers": 0,
            "images": 0,
            "names": [],
            "error": str(exc)[:120],
        }
    _docker_cache = (now, payload)
    return dict(payload)


def _collect_system_snapshot() -> dict[str, Any]:
    gpu = None
    gpu_error = ""
    try:
        gpu = _collect_gpu_snapshot()
    except Exception as exc:
        gpu_error = str(exc)
    host = _collect_host_resources()
    disk_io: dict[str, Any] = {}
    disk_io_error = ""
    try:
        disk_io = _collect_disk_io()
    except Exception as exc:
        disk_io_error = str(exc)
    network: dict[str, Any] = {}
    network_error = ""
    try:
        network = _collect_network()
    except Exception as exc:
        network_error = str(exc)
    docker = _collect_docker()
    return {
        "gpu": gpu,
        "gpu_error": gpu_error,
        "cpu": host["cpu"],
        "ram": host["ram"],
        "swap": host["swap"],
        "disk_io": disk_io,
        "disk_io_error": disk_io_error,
        "network": network,
        "network_error": network_error,
        "docker": docker,
    }


def _make_progress_bar(bar_id: str) -> ProgressBar:
    try:
        return ProgressBar(
            total=100,
            show_eta=False,
            show_percentage=False,
            id=bar_id,
            classes="sys-bar",
        )
    except TypeError:
        return ProgressBar(total=100, show_eta=False, id=bar_id, classes="sys-bar")


class SystemTilePanel(Vertical):
    DEFAULT_CSS = """
    SystemTilePanel {
        border: solid $primary;
        padding: 0 1;
        height: 24;
        min-height: 18;
    }
    SystemTilePanel.tile-span-2 {
        column-span: 2;
    }
    #system-title {
        height: 1;
        text-style: bold;
    }
    #system-cols, #system-extra {
        height: auto;
        layout: horizontal;
        align: left top;
        margin-bottom: 0;
    }
    .sys-col {
        width: 1fr;
        height: auto;
        padding-right: 1;
        layout: vertical;
        overflow: hidden;
    }
    .sys-col-title {
        height: 1;
        max-height: 1;
        text-style: bold;
    }
    .sys-col-meta {
        height: 1;
        max-height: 1;
        color: $text-muted;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sys-bar {
        height: 1;
        max-height: 1;
        width: 100%;
        margin: 0;
        padding: 0;
    }
    .sys-bar-label, .sys-spark {
        height: 1;
        max-height: 1;
        color: $text-muted;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sys-extra-body {
        height: 4;
        max-height: 4;
        color: $text-muted;
        overflow: hidden;
    }
    #system-table {
        height: 1fr;
        min-height: 4;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._gpu_hist: deque[float] = deque(maxlen=_HISTORY_LEN)
        self._cpu_hist: deque[float] = deque(maxlen=_HISTORY_LEN)
        self._ram_hist: deque[float] = deque(maxlen=_HISTORY_LEN)

    def compose(self) -> ComposeResult:
        yield Static("system", id="system-title")
        with Horizontal(id="system-cols"):
            with Vertical(classes="sys-col"):
                yield Static("gpu", classes="sys-col-title")
                yield Static("", id="sys-gpu-meta", classes="sys-col-meta")
                yield _make_progress_bar("sys-gpu-bar")
                yield Static("", id="sys-gpu-bar-label", classes="sys-bar-label")
                yield Static("", id="sys-gpu-spark", classes="sys-spark")
            with Vertical(classes="sys-col"):
                yield Static("cpu", classes="sys-col-title")
                yield Static("", id="sys-cpu-meta", classes="sys-col-meta")
                yield _make_progress_bar("sys-cpu-bar")
                yield Static("", id="sys-cpu-bar-label", classes="sys-bar-label")
                yield Static("", id="sys-cpu-spark", classes="sys-spark")
            with Vertical(classes="sys-col"):
                yield Static("ram", classes="sys-col-title")
                yield Static("", id="sys-ram-meta", classes="sys-col-meta")
                yield _make_progress_bar("sys-ram-bar")
                yield Static("", id="sys-ram-bar-label", classes="sys-bar-label")
                yield Static("", id="sys-ram-spark", classes="sys-spark")
        with Horizontal(id="system-extra"):
            with Vertical(classes="sys-col"):
                yield Static("io", classes="sys-col-title")
                yield Static("", id="sys-io-body", classes="sys-extra-body")
            with Vertical(classes="sys-col"):
                yield Static("net", classes="sys-col-title")
                yield Static("", id="sys-net-body", classes="sys-extra-body")
            with Vertical(classes="sys-col"):
                yield Static("docker", classes="sys-col-title")
                yield Static("", id="sys-docker-body", classes="sys-extra-body")
        yield DataTable(id="system-table")

    def on_mount(self) -> None:
        table = self.query_one("#system-table", DataTable)
        table.add_columns("pid", "process", "vram")
        table.cursor_type = "row"
        try:
            import psutil

            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def apply(self, state: TileState) -> None:
        title = self.query_one("#system-title", Static)
        table = self.query_one("#system-table", DataTable)
        payload = state.payload or {}
        if not state.ok or not payload:
            title.update("system")
            return
        title.update("system")

        gpu = payload.get("gpu") if isinstance(payload.get("gpu"), dict) else None
        gpu_meta = self.query_one("#sys-gpu-meta", Static)
        gpu_bar = self.query_one("#sys-gpu-bar", ProgressBar)
        gpu_label = self.query_one("#sys-gpu-bar-label", Static)
        gpu_spark = self.query_one("#sys-gpu-spark", Static)
        if gpu:
            used = int(gpu.get("used_bytes") or 0)
            total = max(1, int(gpu.get("total_bytes") or 1))
            pct = min(100.0, (used / total) * 100.0)
            used_gib = used / (2**30)
            self._gpu_hist.append(used_gib)
            bits = [str(gpu.get("name") or "gpu"), f"util {gpu.get('gpu_util')}%"]
            if gpu.get("temp_c") is not None:
                bits.append(f"{gpu.get('temp_c')} C")
            if gpu.get("power_w") is not None:
                bits.append(f"{float(gpu['power_w']):.0f} W")
            gpu_meta.update("  ".join(bits))
            gpu_bar.update(progress=pct)
            gpu_label.update(_fmt_pair(used, total))
            gpu_spark.update(sparkline(self._gpu_hist))
            table.clear()
            processes = gpu.get("processes") or []
            if not processes:
                table.add_row("-", "(no compute processes / WSL limited)", "-")
            else:
                for row in processes[:8]:
                    table.add_row(
                        str(row.get("pid")),
                        str(row.get("name")),
                        _fmt_bytes(int(row.get("vram_bytes") or 0)),
                    )
        else:
            gpu_meta.update(str(payload.get("gpu_error") or "gpu unavailable"))
            gpu_bar.update(progress=0)
            gpu_label.update("-")
            gpu_spark.update("")
            table.clear()
            table.add_row("-", "gpu unavailable", "-")

        cpu = payload.get("cpu") if isinstance(payload.get("cpu"), dict) else {}
        cpu_util = float(cpu.get("util") or 0.0)
        self._cpu_hist.append(cpu_util)
        cpu_bits = [f"{int(cpu.get('count') or 0)} thr"]
        if cpu.get("freq_mhz") is not None:
            cpu_bits.append(f"{float(cpu['freq_mhz']):.0f} MHz")
        self.query_one("#sys-cpu-meta", Static).update("  ".join(cpu_bits))
        self.query_one("#sys-cpu-bar", ProgressBar).update(progress=min(100.0, cpu_util))
        self.query_one("#sys-cpu-bar-label", Static).update(f"{cpu_util:.0f}%")
        self.query_one("#sys-cpu-spark", Static).update(sparkline(self._cpu_hist))

        ram = payload.get("ram") if isinstance(payload.get("ram"), dict) else {}
        used = int(ram.get("used_bytes") or 0)
        total = max(1, int(ram.get("total_bytes") or 1))
        pct = float(ram.get("percent") or ((used / total) * 100.0))
        used_gib = used / (2**30)
        self._ram_hist.append(used_gib)
        avail = int(ram.get("avail_bytes") or 0)
        swap = payload.get("swap") if isinstance(payload.get("swap"), dict) else {}
        ram_meta = f"free {_fmt_bytes(avail)}"
        if float(swap.get("percent") or 0) > 0.1:
            ram_meta += f"  swap {float(swap.get('percent') or 0):.0f}%"
        self.query_one("#sys-ram-meta", Static).update(ram_meta)
        self.query_one("#sys-ram-bar", ProgressBar).update(progress=min(100.0, pct))
        self.query_one("#sys-ram-bar-label", Static).update(_fmt_pair(used, total))
        self.query_one("#sys-ram-spark", Static).update(sparkline(self._ram_hist))

        io_body = self.query_one("#sys-io-body", Static)
        disk_io = payload.get("disk_io") if isinstance(payload.get("disk_io"), dict) else {}
        if disk_io.get("ok"):
            io_body.update(
                "\n".join(
                    [
                        f"read  {_fmt_rate(float(disk_io.get('read_bps') or 0))}",
                        f"write {_fmt_rate(float(disk_io.get('write_bps') or 0))}",
                    ]
                )
            )
        else:
            io_body.update(
                str(disk_io.get("error") or payload.get("disk_io_error") or "io n/a")
            )

        net_body = self.query_one("#sys-net-body", Static)
        network = payload.get("network") if isinstance(payload.get("network"), dict) else {}
        if network:
            ifaces = ", ".join(network.get("ifaces") or []) or "n/a"
            net_body.update(
                "\n".join(
                    [
                        f"↓ {_fmt_rate(float(network.get('down_bps') or 0))}",
                        f"↑ {_fmt_rate(float(network.get('up_bps') or 0))}",
                        f"if {ifaces}",
                    ]
                )
            )
        else:
            net_body.update(str(payload.get("network_error") or "net n/a"))

        docker_body = self.query_one("#sys-docker-body", Static)
        docker = payload.get("docker") if isinstance(payload.get("docker"), dict) else {}
        if docker.get("ok"):
            names = ", ".join(docker.get("names") or []) or "-"
            docker_body.update(
                "\n".join(
                    [
                        f"{int(docker.get('containers') or 0)} ctr  "
                        f"{int(docker.get('images') or 0)} img",
                        names,
                    ]
                )
            )
        else:
            docker_body.update(str(docker.get("error") or "docker n/a"))


class SystemTile(BaseTile):
    tile_id = "system"
    refresh_s = 2.0
    span_cols = 2

    def panel(self) -> Any:
        panel = SystemTilePanel(id=f"dash-{self.tile_id}")
        panel.add_class("tile-span-2")
        return panel

    def fetch(self, app: Any) -> TileState:
        try:
            snap = _collect_system_snapshot()
        except Exception as exc:
            return TileState(
                ok=False,
                text=f"system\n{exc}",
                fetched_at=time.time(),
                error=str(exc),
            )
        gpu = snap.get("gpu") or {}
        cpu = snap.get("cpu") or {}
        ram = snap.get("ram") or {}
        lines = ["system"]
        if gpu:
            lines.append(
                f"gpu {gpu.get('name')}  vram {_fmt_bytes(int(gpu['used_bytes']))} / "
                f"{_fmt_bytes(int(gpu['total_bytes']))}  util {gpu.get('gpu_util')}%"
            )
        elif snap.get("gpu_error"):
            lines.append(f"gpu: {snap['gpu_error']}")
        lines.append(f"cpu {cpu.get('util', 0):.0f}%  ({cpu.get('count', 0)} thr)")
        lines.append(
            f"ram {_fmt_bytes(int(ram.get('used_bytes') or 0))} / "
            f"{_fmt_bytes(int(ram.get('total_bytes') or 0))}"
        )
        disk_io = snap.get("disk_io") or {}
        if disk_io.get("ok"):
            lines.append(
                f"io r {_fmt_rate(float(disk_io.get('read_bps') or 0))}  "
                f"w {_fmt_rate(float(disk_io.get('write_bps') or 0))}"
            )
        return TileState(ok=True, text="\n".join(lines), fetched_at=time.time(), payload=snap)
