from __future__ import annotations

import shutil
from pathlib import Path

from training.orchestrator.config import TrainerConfig
from training.orchestrator.hf_lora import HfLoraDriver, nearest_eligible_preset, session_trainable
from training.orchestrator.protocol import CatalogRow
from training.orchestrator.spec import DRIVER_HF_LORA, DRIVER_SSL4SED, TrainerSpec, spec_from_session
from training.orchestrator.ssl4sed import Ssl4sedDriver
from utils.device.env_bootstrap import sophon_project_root


def _fmt_bytes(value: object) -> str:
    if not isinstance(value, int) or value < 0:
        return "unknown"
    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= div:
            return f"{value / div:.2f} {label}"
    return f"{value} B"


def _host_lines(project_root: Path, config: TrainerConfig, ssl: Ssl4sedDriver) -> list[str]:
    lines: list[str] = []
    try:
        from utils.device.system_check import collect_system_snapshot

        snapshot = collect_system_snapshot(device="all")
    except Exception as exc:
        snapshot = {"error": str(exc)}
    host = snapshot.get("host_memory", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(host, dict):
        host = {}
    devices = snapshot.get("devices", []) if isinstance(snapshot, dict) else []
    device_line = "device: cpu"
    if isinstance(devices, list) and devices:
        first = devices[0]
        if isinstance(first, dict) and first.get("kind") == "cuda":
            device_line = (
                f"cuda:{first.get('index')} {first.get('name')} "
                f"free={_fmt_bytes(first.get('free_bytes'))} / {_fmt_bytes(first.get('total_bytes'))}"
            )
        elif isinstance(first, dict):
            device_line = f"{first.get('kind')}: {first.get('name', first.get('detail', 'available'))}"
    ram_line = (
        f"ram={_fmt_bytes(host.get('host_avail_bytes'))} free / "
        f"{_fmt_bytes(host.get('host_total_bytes'))}"
    )
    try:
        usage = shutil.disk_usage(project_root)
        disk_line = f"disk={_fmt_bytes(usage.free)} free / {_fmt_bytes(usage.total)}  (sophon)"
    except OSError:
        disk_line = "disk=unknown  (sophon)"
    total, avail = ssl.host.disk_bytes()
    ssl_disk = (
        f"ssl4sed disk={_fmt_bytes(avail)} free / {_fmt_bytes(total)}  ({config.ssl4sed_root})"
    )
    py = snapshot.get("python") if isinstance(snapshot, dict) else None
    torch = snapshot.get("torch") if isinstance(snapshot, dict) else None
    lines.append(f"trainer  python={py} torch={torch}")
    lines.append(f"         {device_line}")
    lines.append(f"         {ram_line}")
    lines.append(f"         {disk_line}")
    lines.append(f"         {ssl_disk}")
    return lines


def _session_lines(state: object, project_root: Path) -> list[str]:
    backend = str(getattr(state, "backend_id", "") or "")
    preset = getattr(state, "preset_key", None)
    server = getattr(state, "server_model", None)
    model = str(preset or server or getattr(state, "model_path", "") or "-")
    ok, reason = session_trainable(state)
    flag = "yes" if ok else "no"
    lines = [
        f"session  backend={backend}  model={model}",
        f"         trainable={flag}  ({reason})",
    ]
    if not ok:
        nearest, on_disk = nearest_eligible_preset(project_root)
        if nearest:
            mark = "on disk" if on_disk else "weights missing. /model-download " + nearest
            lines.append(f"         hf-lora default would be: {nearest}  [{mark}]")
    return lines


def _catalog_block(rows: list[CatalogRow], *, kind: str, cap: int) -> list[str]:
    picked = [row for row in rows if row.kind == kind][:cap]
    if not picked:
        return [f"  {kind}   (none)"]
    lines: list[str] = []
    for row in picked:
        size = f"  {row.size_hint}" if row.size_hint else ""
        lines.append(f"  {kind:<6} {row.key:<28} {row.status:<10}{size}  {row.detail}")
    return lines


def format_trainer_sheet(state: object) -> str:
    from training.orchestrator.config import load_trainer_config

    project_root = getattr(state, "project_root", None) or sophon_project_root()
    config = load_trainer_config(project_root)
    spec = spec_from_session(state)
    hf = HfLoraDriver(state)
    ssl = Ssl4sedDriver(state, config)
    hf_rows = hf.inventory()
    ssl_rows: list[CatalogRow] = []
    ssl_note = "ready-code"
    try:
        ssl_rows = ssl.inventory()
    except Exception as exc:
        ssl_note = f"unreachable ({exc})"
    default_target = spec.target or ssl.default_target()
    lines: list[str] = []
    lines.extend(_host_lines(project_root, config, ssl))
    lines.append("")
    lines.extend(_session_lines(state, project_root))
    lines.append("")
    lines.append("subtrainers")
    lines.append(f"  {DRIVER_HF_LORA:<10} host=windows-gpu  extra=finetune")
    lines.append(
        f"  {DRIVER_SSL4SED:<10} host=linux/{config.wsl_distro}  {ssl_note}  target={default_target}"
    )
    lines.append(f"  active={spec.driver}  task={spec.task or '-'}  data={spec.data or '-'}  model={spec.model or '-'}  target={spec.target or default_target}")
    lines.append("")
    lines.append("catalog (capped)")
    lines.extend(_catalog_block(hf_rows, kind="task", cap=2))
    lines.extend(_catalog_block(ssl_rows, kind="task", cap=3))
    lines.extend(_catalog_block(hf_rows, kind="data", cap=3))
    lines.extend(_catalog_block(ssl_rows, kind="data", cap=3))
    lines.extend(_catalog_block(hf_rows, kind="model", cap=4))
    lines.append("")
    lines.append("examples")
    model_ex = spec.model or (nearest_eligible_preset(project_root)[0] or "llama2_7b_chat")
    data_ex = spec.data or "gsm8k_instructions"
    lines.append(f"  /trainer select task=instruction-sft data={data_ex} model={model_ex}")
    lines.append(f"  /trainer data get {data_ex}")
    lines.append("  /trainer run")
    lines.append("  /trainer use ssl4sed")
    sed_key = "sed.desed.synthetic_v4"
    for row in ssl_rows:
        if row.kind == "data" and row.key.startswith("sed."):
            sed_key = row.key
            break
    lines.append(f"  /trainer data get {sed_key}")
    lines.append(f"  /trainer run target={default_target}")
    lines.append("  /trainer run full")
    lines.append("  /trainer watch")
    return "\n".join(lines)
