from __future__ import annotations

import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)


import argparse
import ctypes
import json
import os
import platform
import re
import subprocess
from typing import Any

from backend.hf.backend import mps_ready
from backend.hf.paths import infer_default_quantization, resolve_local_model_dir
from utils.device.env_bootstrap import load_orodruin_dotenv


def _fmt_bytes(n: int) -> str:
    if n < 0:
        return "unknown"
    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= div:
            return f"{n / div:.2f} {label}"
    return f"{n} B"


def _linux_mem_available() -> int | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _windows_mem_status() -> tuple[int | None, int | None]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32 = ctypes.windll.kernel32
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None, None
    total = int(stat.ullTotalPhys)
    avail = int(stat.ullAvailPhys)
    return total, avail


def _macos_mem() -> tuple[int | None, int | None]:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    if not out.isdigit():
        return None, None
    total = int(out)
    try:
        vm = subprocess.check_output(["vm_stat"], stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.CalledProcessError):
        return total, None
    page = 4096
    m = re.search(r"page size of (\d+) bytes", vm)
    if m:
        page = int(m.group(1))
    free_pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m2 = re.search(rf"{label}:\s+(\d+)\.", vm)
        if m2:
            free_pages += int(m2.group(1))
    approx_avail = free_pages * page
    return total, approx_avail


def _host_memory_snapshot() -> dict[str, Any]:
    system = platform.system()
    total: int | None = None
    avail: int | None = None
    detail = ""
    if system == "Linux":
        avail = _linux_mem_available()
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        total = int(parts[1]) * 1024
                    break
        except OSError:
            pass
        detail = "MemTotal/MemAvailable from /proc/meminfo"
    elif system == "Darwin":
        total, avail = _macos_mem()
        detail = "hw.memsize and vm_stat rough free+inactive (not exactAvail)"
    elif system == "Windows":
        total, avail = _windows_mem_status()
        detail = "GlobalMemoryStatusEx ullTotalPhys/ullAvailPhys"
    return {
        "platform": system,
        "host_total_bytes": total,
        "host_avail_bytes": avail,
        "host_detail": detail,
    }


def _model_weight_bytes_hint(model_dir: Path) -> int | None:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        meta = data.get("metadata") or {}
        ts = meta.get("total_size")
        if isinstance(ts, int) and ts > 0:
            return ts
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return None


def _report_cuda_line(i: int) -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(i)
    name = props.name
    total = int(props.total_memory)
    free: int | None = None
    used: int | None = None
    try:
        free, _total2 = torch.cuda.mem_get_info(i)
        used = total - int(free)
    except Exception:
        pass
    cap = torch.cuda.get_device_capability(i)
    return {
        "kind": "cuda",
        "index": i,
        "name": name,
        "total_bytes": total,
        "free_bytes": free,
        "used_bytes": used,
        "capability": f"sm_{cap[0]}{cap[1]}",
    }


def _bitsandbytes_note() -> str:
    try:
        import bitsandbytes as bnb  # noqa: F401

        _ = bnb
        return "import_ok"
    except Exception as exc:
        return f"import_failed ({type(exc).__name__}: {exc})"


def _allocate_probe(device: str, megabytes: int) -> dict[str, Any]:
    import torch

    elements = max(1, (megabytes * 1024 * 1024) // 4)
    out: dict[str, Any] = {"device": device, "requested_mib": megabytes, "ok": False, "error": None}
    try:
        t = torch.empty((elements,), dtype=torch.float32, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        out["ok"] = True
        del t
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except RuntimeError as exc:
        out["error"] = str(exc)
    return out


def _choose_recommended_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if mps_ready():
        return "mps"
    return "cpu"


def collect_system_snapshot(
    *,
    device: str = "all",
    cuda_device: int = 0,
    model: str | None = None,
) -> dict[str, Any]:
    import torch

    host = _host_memory_snapshot()
    recommended = _choose_recommended_device()
    row: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_ready_orodruin": bool(mps_ready()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gemma4_quantization_default": infer_default_quantization(),
        "host_memory": host,
        "bitsandbytes": _bitsandbytes_note() if torch.cuda.is_available() else "skipped_no_cuda",
        "devices": [],
        "recommended_device": recommended,
    }

    if model:
        mp = resolve_local_model_dir(model)
        row["model_dir"] = str(mp)
        row["model_config_present"] = (mp / "config.json").is_file()
        row["model_weight_index_total_bytes"] = _model_weight_bytes_hint(mp)

    dev_rows: list[dict[str, Any]] = []
    mode = device
    if mode == "auto":
        if recommended.startswith("cuda") and torch.cuda.is_available():
            cidx = cuda_device
            if ":" in recommended:
                try:
                    cidx = int(recommended.split(":", 1)[1])
                except ValueError:
                    cidx = cuda_device
            if 0 <= cidx < torch.cuda.device_count():
                dev_rows.append(_report_cuda_line(cidx))
        elif recommended == "mps" and mps_ready():
            dev_rows.append({"kind": "mps", "index": 0, "name": "Apple MPS", "detail": "no_free_mem_api"})
        else:
            dev_rows.append({"kind": "cpu", "index": 0, "name": platform.processor() or "cpu"})
    else:
        if mode in ("all", "cuda") and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                dev_rows.append(_report_cuda_line(i))
        if mode in ("all", "mps") and mps_ready():
            dev_rows.append({"kind": "mps", "index": 0, "name": "Apple MPS", "detail": "no_free_mem_api"})
        if mode in ("all", "cpu") or (mode == "all" and not dev_rows):
            dev_rows.append({"kind": "cpu", "index": 0, "name": platform.processor() or "cpu"})

    row["devices"] = dev_rows
    return row


def main() -> None:
    load_orodruin_dotenv()
    parser = argparse.ArgumentParser(
        description="Host and PyTorch device checks for orodruin (CUDA, MPS, CPU, memory hints).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu", "all"),
        default="all",
        help="Which backends to describe; auto prints a single recommended line.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index for cuda-specific probes.",
    )
    parser.add_argument(
        "--allocate-mib",
        type=int,
        default=0,
        metavar="N",
        help="If >0, allocate N MiB of float32 on the probe device once (catches many OOM paths).",
    )
    parser.add_argument(
        "--probe-device",
        default=None,
        metavar="SPEC",
        help="Device string for --allocate-mib (e.g. cuda:0, mps, cpu). Default: recommended auto device.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional local model dir; reports index total_size vs free memory when known.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object on stdout instead of human text.",
    )
    args = parser.parse_args()

    row = collect_system_snapshot(device=args.device, cuda_device=args.cuda_device, model=args.model)
    dev_rows = list(row.get("devices", []))
    host = row["host_memory"]

    probe_target = args.probe_device or row["recommended_device"]
    if args.allocate_mib > 0:
        row["allocate_probe"] = _allocate_probe(probe_target, args.allocate_mib)
        row["allocate_probe_target"] = probe_target

    if row.get("model_weight_index_total_bytes") is not None and dev_rows:
        first = dev_rows[0]
        free_b = first.get("free_bytes")
        need_b = row["model_weight_index_total_bytes"]
        if isinstance(free_b, int) and isinstance(need_b, int):
            row["model_fits_cuda0_free_guess"] = free_b >= need_b

    if args.json:
        print(json.dumps(row, indent=2))
        return

    print("orodruin system check")
    print(f"  python={row['python']} platform={row['platform']}")
    print(f"  torch={row['torch']} torch.version.cuda={row['torch_cuda_build']}")
    print(f"  cuda_available={row['cuda_available']} count={row['cuda_device_count']}")
    print(
        f"  mps built={row['mps_built']} available={row['mps_available']} "
        f"orodruin_mps_ready={row['mps_ready_orodruin']}"
    )
    vis = row["cuda_visible_devices"]
    if vis:
        print(f"  CUDA_VISIBLE_DEVICES={vis!r}")
    print(f"  infer_default_quantization() -> {row['gemma4_quantization_default']!r}")
    print(f"  bitsandbytes: {row['bitsandbytes']}")
    ht, ha = host.get("host_total_bytes"), host.get("host_avail_bytes")
    print(
        f"  host RAM total={_fmt_bytes(ht) if isinstance(ht, int) else 'unknown'} "
        f"avail~={_fmt_bytes(ha) if isinstance(ha, int) else 'unknown'} ({host.get('host_detail', '')})"
    )
    print(f"  recommended_device={row['recommended_device']!r}")
    for d in dev_rows:
        if d.get("kind") == "cuda":
            print(
                f"  cuda:{d['index']} {d.get('name')} "
                f"total={_fmt_bytes(int(d['total_bytes']))} "
                f"free~={_fmt_bytes(int(d['free_bytes'])) if d.get('free_bytes') is not None else 'unknown'} "
                f"cap={d.get('capability')}"
            )
        elif d.get("kind") == "mps":
            print(f"  mps: {d.get('detail')}")
        else:
            print(f"  cpu: {d.get('name')}")
    if args.model:
        print(f"  model_dir={row.get('model_dir')}")
        print(f"  config.json present={row.get('model_config_present')}")
        w = row.get("model_weight_index_total_bytes")
        if isinstance(w, int):
            print(f"  model.safetensors.index total_size={_fmt_bytes(w)}")
            if "model_fits_cuda0_free_guess" in row:
                print(f"  model fits first cuda free report={row['model_fits_cuda0_free_guess']}")
        else:
            print("  model.safetensors.index total_size=unknown (missing or unreadable)")
    if args.allocate_mib > 0:
        ap = row.get("allocate_probe") or {}
        print(
            f"  allocate_probe device={row.get('allocate_probe_target')!r} "
            f"mib={args.allocate_mib} ok={ap.get('ok')} err={ap.get('error')!r}"
        )



# TODO(utils.device.system_check): optional nvml-based VRAM via pynvml when users install extras for richer telemetry


if __name__ == "__main__":
    main()
