from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.chat_resolve import list_server_models, lmstudio_reachable, ollama_reachable
from backend.hf.registry import (
    HF_MODEL_PRESETS,
    local_only_model_dirs,
    preset_has_weights,
    resolve_preset_dir,
)
from backend.providers import is_server_backend

SCORE_RECIPE = (
    "local before download, then score = 2*tok/s + 5*log10(ctx) + 30*local + 15*tools[agent] - 10*n_deps"
    "  (CPU offload -100 after VRAM hard filter, 10% headroom)"
)

_HEADROOM = 0.10
_CTX_LADDER = (4096, 8192, 16384, 32768)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)[bB]")
_NON_CHAT_RE = re.compile(
    r"(embed(?:ding)?|nomic-embed|rerank|whisper|\btts\b|\basr\b|vocoder|\bclip\b)",
    re.IGNORECASE,
)
_INSTRUCT_MARKERS = ("_it", "instruct", "chat", "qat", "-it")
_QUANT_BYTES = {
    "q4": 0.55,
    "4bit": 0.55,
    "q8": 1.0,
    "8bit": 1.0,
    "fp16": 2.0,
    "none": 2.0,
    "server-gguf": 0.55,
}


@dataclass
class DeviceSnapshot:
    cuda: bool
    gpu_name: str
    vram_total_bytes: int
    vram_free_bytes: int
    ram_total_bytes: int
    ram_avail_bytes: int
    disk_free_bytes: int
    bitsandbytes_ok: bool
    default_quant: str
    recommended_device: str


@dataclass
class SetupCombo:
    backend: str
    model_id: str
    quant: str
    cli_quant: str
    ctx: int
    model_max_ctx: int
    params_b: float | None
    tok_s: float
    tok_s_source: str
    disk_bytes: int | None
    disk_local: bool
    vram_weights_gb: float
    vram_kv_gb: float
    vram_need_gb: float
    headroom_gb: float
    fits: bool
    tools: bool
    missing_deps: tuple[str, ...]
    download_hint: str
    cpu_offload: bool
    score: float = 0.0
    current: bool = False
    lmstudio_listed: bool = True
    max_new_tokens: int = 512

    def switch_target(self) -> str:
        if self.backend == "hf":
            return self.model_id
        return f"{self.backend}:{self.model_id}"


@dataclass
class SetupReport:
    device: DeviceSnapshot
    recipe: str
    ranked: list[SetupCombo]
    shown: list[SetupCombo]
    current: SetupCombo | None
    table: str
    pending: dict[str, Any] | None
    scored_n: int = 0
    skipped_n: int = 0


def combo_to_dict(combo: SetupCombo) -> dict[str, Any]:
    row = asdict(combo)
    row["missing_deps"] = list(combo.missing_deps)
    return row


def combo_from_dict(row: dict[str, Any]) -> SetupCombo:
    data = dict(row)
    data["missing_deps"] = tuple(data.get("missing_deps") or ())
    return SetupCombo(**data)


def format_gib(nbytes: int | None, digits: int = 1) -> str:
    if nbytes is None:
        return "-"
    return f"{nbytes / (1024 ** 3):.{digits}f}G"


def collect_device_snapshot(project_root: Path) -> DeviceSnapshot:
    snap: dict[str, Any]
    try:
        from utils.device.system_check import collect_system_snapshot

        snap = collect_system_snapshot(device="all")
    except Exception:
        snap = {
            "cuda_available": False,
            "devices": [],
            "host_memory": {},
            "gemma4_quantization_default": "none",
            "recommended_device": "cpu",
            "bitsandbytes": "skipped",
        }
    cuda_row: dict[str, Any] | None = None
    for row in snap.get("devices") or []:
        if row.get("kind") == "cuda":
            cuda_row = row
            break
    host = snap.get("host_memory") or {}
    models_dir = project_root / "models"
    disk_target = models_dir if models_dir.exists() else project_root
    try:
        disk_free = int(shutil.disk_usage(str(disk_target)).free)
    except OSError:
        disk_free = 0
    bnb = str(snap.get("bitsandbytes") or "")
    return DeviceSnapshot(
        cuda=bool(snap.get("cuda_available") and cuda_row is not None),
        gpu_name=str((cuda_row or {}).get("name") or ""),
        vram_total_bytes=int((cuda_row or {}).get("total_bytes") or 0),
        vram_free_bytes=int((cuda_row or {}).get("free_bytes") or 0),
        ram_total_bytes=int(host.get("host_total_bytes") or 0),
        ram_avail_bytes=int(host.get("host_avail_bytes") or 0),
        disk_free_bytes=disk_free,
        bitsandbytes_ok=bnb == "import_ok",
        default_quant=str(snap.get("gemma4_quantization_default") or "none"),
        recommended_device=str(snap.get("recommended_device") or "cpu"),
    )


def infer_params_b(text: str, vram_class: str = "") -> float | None:
    blob = f"{text} {vram_class}".lower()
    if "e2b" in blob:
        return 5.0
    if "e4b" in blob:
        return 8.0
    if "a4b" in blob or "26b" in blob:
        return 26.0
    if vram_class:
        match = _SIZE_RE.search(vram_class)
        if match:
            return float(match.group(1))
    match = _SIZE_RE.search(blob.replace("_", "-"))
    if match:
        return float(match.group(1))
    return None


def is_non_chat_name(text: str) -> bool:
    blob = str(text or "").replace("_", "-")
    return bool(_NON_CHAT_RE.search(blob))


def _is_instructish(key: str) -> bool:
    blob = str(key or "").lower()
    return any(marker in blob for marker in _INSTRUCT_MARKERS)


def has_chat_sibling(key: str, keys: set[str]) -> bool:
    if _is_instructish(key):
        return False
    if key.endswith("_base") or key.endswith("_pretrained"):
        stem = key.rsplit("_", 1)[0]
        candidates = (stem, f"{stem}_it", f"{stem}_instruct", f"{stem}_chat")
        return any(item in keys and item != key for item in candidates)
    return any(f"{key}{suffix}" in keys for suffix in ("_it", "_instruct", "_chat"))


def infer_model_max_ctx(text: str) -> int:
    blob = text.lower()
    if "gemma4" in blob or "gemma-4" in blob:
        return 32768
    if "llama-3.1" in blob or "llama3_1" in blob or "llama3.1" in blob:
        return 131072
    if "llama-3" in blob or "llama3" in blob:
        return 8192
    if "qwen" in blob:
        return 32768
    return 8192


def infer_quant_label(text: str, cli_quant: str = "") -> str:
    blob = text.lower()
    if cli_quant in ("4bit", "q4"):
        return "q4"
    if cli_quant in ("8bit", "q8"):
        return "q8"
    if cli_quant in ("none", "fp16"):
        return "fp16"
    if "q8" in blob or "8bit" in blob or "int8" in blob:
        return "q8"
    if "q4" in blob or "4bit" in blob or "qat" in blob or "gguf" in blob:
        return "q4"
    if "fp16" in blob or "f16" in blob:
        return "fp16"
    return "server-gguf"


def bytes_per_param(quant: str) -> float:
    return _QUANT_BYTES.get(quant, 0.55)


def estimate_weight_gb(params_b: float | None, quant: str) -> float:
    if params_b is None or params_b <= 0:
        return 0.0
    return params_b * bytes_per_param(quant)


def estimate_kv_gb(params_b: float | None, ctx: int, quant: str) -> float:
    if params_b is None or params_b <= 0:
        return 0.0
    kv_scale = 0.16 if quant in ("fp16", "none") else 0.08
    return params_b * (ctx / 4096.0) * kv_scale


def estimate_tok_s(
    params_b: float | None,
    quant: str,
    device: DeviceSnapshot,
    *,
    measured: float | None = None,
) -> tuple[float, str]:
    if measured is not None and measured > 0:
        return float(measured), "measured"
    if params_b is None or params_b <= 0:
        return 0.0, "unknown"
    if not device.cuda:
        return max(1.5, 12.0 / max(params_b, 1.0)), "est"
    base = 90.0 / max(params_b, 1.0)
    if quant in ("q4", "4bit", "server-gguf"):
        base *= 1.6
    elif quant in ("q8", "8bit"):
        base *= 1.15
    name = device.gpu_name.lower()
    if "4090" in name:
        base *= 1.3
    elif "3090" in name:
        base *= 1.0
    elif "3080" in name:
        base *= 0.75
    return max(3.0, min(base, 140.0)), "est"


def _dir_bytes(path: Path) -> int | None:
    if not path.is_dir():
        return None
    total = 0
    found = False
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
                found = True
            except OSError:
                continue
    if not found:
        return None
    return total


def _largest_fitting_ctx(
    params_b: float | None,
    quant: str,
    model_max: int,
    vram_total_gb: float,
    *,
    cpu: bool,
    assume_resident: bool = False,
) -> tuple[int, bool, float, float, float]:
    if params_b is None or params_b <= 0:
        ctx = min(model_max, 8192) if model_max > 0 else 8192
        if assume_resident and not cpu:
            return ctx, True, 0.0, 0.0, vram_total_gb
        return ctx, False, 0.0, 0.0, -0.0
    budget = vram_total_gb * (1.0 - _HEADROOM) if not cpu and vram_total_gb > 0 else 0.0
    weights = estimate_weight_gb(params_b, quant)
    chosen = _CTX_LADDER[0]
    chosen_kv = estimate_kv_gb(params_b, chosen, quant)
    fits = False
    for ctx in _CTX_LADDER:
        if ctx > model_max:
            break
        kv = estimate_kv_gb(params_b, ctx, quant)
        need = weights + kv
        if cpu:
            chosen, chosen_kv, fits = ctx, kv, False
            continue
        if need <= budget:
            chosen, chosen_kv, fits = ctx, kv, True
            continue
        break
    headroom = (vram_total_gb - (weights + chosen_kv)) if vram_total_gb > 0 else -weights
    return chosen, fits, weights, chosen_kv, headroom


def _missing_deps(cli_quant: str, device: DeviceSnapshot, backend: str) -> tuple[str, ...]:
    if backend != "hf":
        return ()
    if cli_quant in ("4bit", "8bit") and device.cuda and not device.bitsandbytes_ok:
        return ("bitsandbytes",)
    return ()


def _score_combo(combo: SetupCombo, *, agent: bool) -> float:
    ctx = max(combo.ctx, 2)
    score = (2.0 * combo.tok_s) + (5.0 * math.log10(ctx))
    if combo.disk_local:
        score += 30.0
    if agent and combo.tools:
        score += 15.0
    score -= 10.0 * len(combo.missing_deps)
    if combo.cpu_offload:
        score -= 100.0
    if combo.params_b is None:
        score -= 50.0
    return score


def _measured_tok_s(state: Any, backend: str, model_id: str) -> float | None:
    if getattr(state, "backend_id", None) != backend:
        return None
    if backend == "hf" and getattr(state, "preset_key", None) != model_id:
        return None
    if backend != "hf" and getattr(state, "server_model", None) != model_id:
        return None
    stats = getattr(state, "stats", None)
    ema = getattr(stats, "ema_tok_s", None) if stats is not None else None
    if ema is None or ema <= 0:
        return None
    return float(ema)


def _is_current(state: Any, combo: SetupCombo) -> bool:
    if getattr(state, "backend_id", None) != combo.backend:
        return False
    if combo.backend == "hf":
        if getattr(state, "preset_key", None) != combo.model_id:
            return False
        return str(getattr(state, "quantization", "none") or "none") == combo.cli_quant
    return getattr(state, "server_model", None) == combo.model_id


def _current_combo(state: Any, device: DeviceSnapshot, *, agent: bool) -> SetupCombo:
    backend = str(getattr(state, "backend_id", "hf") or "hf")
    if backend == "hf":
        model_id = str(getattr(state, "preset_key", None) or getattr(state, "model_path", "") or "(none)")
        cli_quant = str(getattr(state, "quantization", "none") or "none")
        quant = infer_quant_label(model_id, cli_quant)
    else:
        model_id = str(getattr(state, "server_model", None) or "(none)")
        cli_quant = "none"
        quant = infer_quant_label(model_id, "server-gguf")
    params_b = infer_params_b(model_id)
    model_max = infer_model_max_ctx(model_id)
    vram_gb = device.vram_total_bytes / (1024 ** 3) if device.vram_total_bytes else 0.0
    cpu = not device.cuda
    ctx, fits, weights, kv, headroom = _largest_fitting_ctx(
        params_b, quant, model_max, vram_gb, cpu=cpu
    )
    tok_s, source = estimate_tok_s(
        params_b, quant, device, measured=_measured_tok_s(state, backend, model_id)
    )
    combo = SetupCombo(
        backend=backend,
        model_id=model_id,
        quant=quant,
        cli_quant=cli_quant if backend == "hf" else "none",
        ctx=ctx,
        model_max_ctx=model_max,
        params_b=params_b,
        tok_s=tok_s,
        tok_s_source=source,
        disk_bytes=None,
        disk_local=True,
        vram_weights_gb=weights,
        vram_kv_gb=kv,
        vram_need_gb=weights + kv,
        headroom_gb=headroom,
        fits=fits,
        tools=True,
        missing_deps=_missing_deps(cli_quant, device, backend),
        download_hint="",
        cpu_offload=cpu or not fits,
        current=True,
        max_new_tokens=min(2048, max(256, ctx // 4)),
    )
    combo.score = _score_combo(combo, agent=agent)
    return combo


def _hf_candidates(state: Any, device: DeviceSnapshot, *, agent: bool) -> tuple[list[SetupCombo], int]:
    root = Path(getattr(state, "project_root", Path.cwd()))
    vram_gb = device.vram_total_bytes / (1024 ** 3) if device.vram_total_bytes else 0.0
    cpu = not device.cuda
    out: list[SetupCombo] = []
    skipped = 0
    keys = list(HF_MODEL_PRESETS.keys())
    for extra in local_only_model_dirs(root):
        keys.append(str(extra))
    preset_keys = set(HF_MODEL_PRESETS.keys())
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        preset = HF_MODEL_PRESETS.get(key)
        label = key if preset is None else f"{key} {preset.repo_id} {preset.description} {preset.vram_class}"
        if is_non_chat_name(label):
            skipped += 1
            continue
        if key in preset_keys and has_chat_sibling(key, preset_keys):
            skipped += 1
            continue
        params_b = infer_params_b(label, preset.vram_class if preset is not None else "")
        model_max = infer_model_max_ctx(label)
        local = False
        disk_bytes: int | None = None
        download_hint = ""
        if preset is not None:
            local = preset_has_weights(key, root)
            disk_bytes = _dir_bytes(resolve_preset_dir(key, root))
            if not local:
                # TODO: Hub size API for download GiB when weights are missing
                est = int(estimate_weight_gb(params_b, "fp16") * (1024 ** 3))
                download_hint = f"download ~{format_gib(est)}" if params_b else "download"
        else:
            path = Path(key)
            local = path.is_dir()
            disk_bytes = _dir_bytes(path) if local else None
        quants: tuple[tuple[str, str], ...]
        size = params_b or 0.0
        if cpu or params_b is None:
            quants = (("q4", "4bit"),)
        elif size >= 20:
            quants = (("q4", "4bit"),)
        elif size >= 10:
            quants = (("q4", "4bit"), ("fp16", "none"))
        else:
            quants = (("fp16", "none"), ("q4", "4bit"), ("q8", "8bit"))
        for quant, cli_quant in quants:
            ctx, fits, weights, kv, headroom = _largest_fitting_ctx(
                params_b, quant, model_max, vram_gb, cpu=cpu
            )
            tok_s, source = estimate_tok_s(
                params_b, quant, device, measured=_measured_tok_s(state, "hf", key)
            )
            combo = SetupCombo(
                backend="hf",
                model_id=key,
                quant=quant,
                cli_quant=cli_quant,
                ctx=ctx,
                model_max_ctx=model_max,
                params_b=params_b,
                tok_s=tok_s,
                tok_s_source=source,
                disk_bytes=disk_bytes,
                disk_local=local,
                vram_weights_gb=weights,
                vram_kv_gb=kv,
                vram_need_gb=weights + kv,
                headroom_gb=headroom,
                fits=fits,
                tools=True,
                missing_deps=_missing_deps(cli_quant, device, "hf"),
                download_hint=download_hint if not local else "",
                cpu_offload=cpu or not fits,
                max_new_tokens=min(2048, max(256, ctx // 4)),
            )
            combo.current = _is_current(state, combo)
            combo.score = _score_combo(combo, agent=agent)
            out.append(combo)
    return out, skipped


def _server_candidates(
    state: Any,
    device: DeviceSnapshot,
    backend: str,
    *,
    agent: bool,
) -> tuple[list[SetupCombo], int]:
    if backend == "lmstudio" and not lmstudio_reachable():
        return [], 0
    if backend == "ollama" and not ollama_reachable():
        return [], 0
    try:
        names = list_server_models(backend)
    except Exception:
        names = []
    vram_gb = device.vram_total_bytes / (1024 ** 3) if device.vram_total_bytes else 0.0
    cpu = not device.cuda
    out: list[SetupCombo] = []
    skipped = 0
    for name in names:
        if is_non_chat_name(name):
            skipped += 1
            continue
        quant = infer_quant_label(name, "server-gguf")
        params_b = infer_params_b(name)
        model_max = infer_model_max_ctx(name)
        ctx, fits, weights, kv, headroom = _largest_fitting_ctx(
            params_b,
            quant,
            model_max,
            vram_gb,
            cpu=cpu,
            assume_resident=params_b is None,
        )
        tok_s, source = estimate_tok_s(
            params_b, quant, device, measured=_measured_tok_s(state, backend, name)
        )
        combo = SetupCombo(
            backend=backend,
            model_id=name,
            quant=quant if quant != "fp16" else "server-gguf",
            cli_quant="none",
            ctx=ctx,
            model_max_ctx=model_max,
            params_b=params_b,
            tok_s=tok_s,
            tok_s_source=source,
            disk_bytes=None,
            disk_local=True,
            vram_weights_gb=weights,
            vram_kv_gb=kv,
            vram_need_gb=weights + kv,
            headroom_gb=headroom,
            fits=fits,
            tools=True,
            missing_deps=(),
            download_hint="",
            cpu_offload=cpu or not fits,
            lmstudio_listed=True,
            max_new_tokens=min(2048, max(256, ctx // 4)),
        )
        combo.current = _is_current(state, combo)
        combo.score = _score_combo(combo, agent=agent)
        out.append(combo)
    return out, skipped


def _harness_agent(state: Any) -> bool:
    harness = getattr(state, "harness", None)
    mode = getattr(harness, "mode", None) if harness is not None else None
    return str(mode or "").lower() == "agent"


def format_setup_table(report_or_rows: SetupReport | list[SetupCombo], device: DeviceSnapshot | None = None) -> str:
    if isinstance(report_or_rows, SetupReport):
        rows = report_or_rows.shown
        device = report_or_rows.device
        recipe = report_or_rows.recipe
    else:
        rows = report_or_rows
        recipe = SCORE_RECIPE
    gpu = device.gpu_name if device is not None else ""
    vram = format_gib(device.vram_total_bytes if device is not None else None, 0)
    lines = [
        f"device: {gpu or 'cpu'}  VRAM {vram}  recipe: {recipe}",
        f"{'#':>2} {'backend':<9} {'model':<28} {'quant':<11} {'tok/s':>6} {'disk':<12} {'vram':>6} {'ctx':>6} {'fit':>6} {'local':<8} {'tools':<5} {'deps'}",
    ]
    for index, combo in enumerate(rows, start=1):
        mark = "*" if combo.current else " "
        rank = str(index)
        if combo.current and index > 10:
            rank = "c"
        disk = "local" if combo.disk_local else (combo.download_hint or "download")
        if combo.disk_local and combo.disk_bytes:
            disk = format_gib(combo.disk_bytes)
        if len(disk) > 12:
            disk = disk[:12]
        if combo.tok_s_source == "unknown":
            tok = "-"
        else:
            tok = f"{combo.tok_s:.0f}{'' if combo.tok_s_source == 'est' else '*'}"
        vram_need = f"{combo.vram_need_gb:.1f}G"
        fit = f"{combo.headroom_gb:+.1f}G" if combo.fits else "no"
        local = "yes" if combo.disk_local else "download"
        tools = "yes" if combo.tools else "no"
        deps = ",".join(combo.missing_deps) if combo.missing_deps else "-"
        model = combo.model_id
        if len(model) > 28:
            model = model[:25] + "..."
        lines.append(
            f"{mark}{rank:>2} {combo.backend:<9} {model:<28} {combo.quant:<11} {tok:>6} {disk:<12} {vram_need:>6} {combo.ctx:>6} {fit:>6} {local:<8} {tools:<5} {deps}"
        )
    lines.append("rank 1 is the recommend. * current. tok/s * = measured this session.")
    return "\n".join(lines)


def build_setup_report(state: Any) -> SetupReport:
    root = Path(getattr(state, "project_root", Path.cwd()))
    device = collect_device_snapshot(root)
    agent = _harness_agent(state)
    hf_rows, skip_hf = _hf_candidates(state, device, agent=agent)
    combos: list[SetupCombo] = list(hf_rows)
    skipped = skip_hf
    for backend in ("lmstudio", "ollama"):
        if is_server_backend(backend):
            rows, skip_n = _server_candidates(state, device, backend, agent=agent)
            combos.extend(rows)
            skipped += skip_n
    current = _current_combo(state, device, agent=agent)
    fitting = [row for row in combos if row.fits and not row.cpu_offload]
    pool = fitting if fitting else [row for row in combos if row.cpu_offload]
    ranked = sorted(
        pool,
        key=lambda row: (row.disk_local, row.params_b is not None, row.score),
        reverse=True,
    )
    shown = ranked[:10]
    if current is not None:
        present = any(
            row.backend == current.backend and row.model_id == current.model_id and row.quant == current.quant
            for row in shown
        )
        if not present:
            shown.append(current)
    table = format_setup_table(shown, device)
    pending = combo_to_dict(ranked[0]) if ranked else None
    header = [
        f"read: cuda={device.cuda} gpu={device.gpu_name or '-'} "
        f"vram={format_gib(device.vram_total_bytes)} ram={format_gib(device.ram_total_bytes)} "
        f"disk_free={format_gib(device.disk_free_bytes)} bnb={'ok' if device.bitsandbytes_ok else 'missing'}",
        f"scored {len(combos)} chat / skipped {skipped} non-chat",
    ]
    if not fitting:
        header.append("no CUDA-fit candidate. table is CPU-offload rank (below any CUDA-fit).")
    body = "\n".join(header) + "\n" + table
    return SetupReport(
        device=device,
        recipe=SCORE_RECIPE,
        ranked=ranked,
        shown=shown,
        current=current,
        table=body,
        pending=pending,
        scored_n=len(combos),
        skipped_n=skipped,
    )
