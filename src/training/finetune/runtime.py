from __future__ import annotations

from pathlib import Path

from training.finetune.recipe import FinetuneRecipe


OOM_HINT = (
    "CUDA OOM. Try batch_size=1, raise gradient_accumulation_steps, "
    "lower max_seq_length, lower max_examples, keep load_in_4bit=true."
)


def is_cuda_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _fmt_bytes(n: int | None) -> str:
    if n is None or n < 0:
        return "unknown"
    for label, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= div:
            return f"{n / div:.2f} {label}"
    return f"{n} B"


def preflight_finetune(
    *,
    model_dir: Path,
    recipe: FinetuneRecipe,
) -> list[str]:
    from utils.device.system_check import collect_system_snapshot

    snapshot = collect_system_snapshot(device="cuda", model=str(model_dir))
    lines: list[str] = []
    if not snapshot.get("cuda_available"):
        raise RuntimeError("finetune requires CUDA (torch.cuda.is_available() must be True)")
    bnb = snapshot.get("bitsandbytes")
    if recipe.load_in_4bit and isinstance(bnb, str) and bnb.startswith("import_failed"):
        raise RuntimeError(f"load_in_4bit=true but bitsandbytes is unavailable: {bnb}")
    host = snapshot.get("host_memory") or {}
    host_avail = host.get("host_avail_bytes") if isinstance(host, dict) else None
    lines.append(
        f"preflight host_ram_avail={_fmt_bytes(host_avail if isinstance(host_avail, int) else None)}"
    )
    devices = snapshot.get("devices") or []
    if isinstance(devices, list):
        for row in devices:
            if not isinstance(row, dict) or row.get("kind") != "cuda":
                continue
            idx = row.get("index")
            name = row.get("name")
            free_b = row.get("free_bytes")
            total_b = row.get("total_bytes")
            lines.append(
                f"preflight cuda:{idx} {name} "
                f"free={_fmt_bytes(free_b if isinstance(free_b, int) else None)} "
                f"total={_fmt_bytes(total_b if isinstance(total_b, int) else None)}"
            )
    lines.append(
        "preflight knobs "
        f"batch={recipe.per_device_train_batch_size} "
        f"accum={recipe.gradient_accumulation_steps} "
        f"max_seq={recipe.max_seq_length} "
        f"4bit={recipe.load_in_4bit}"
    )
    weight_hint = snapshot.get("model_weight_index_total_bytes")
    if isinstance(weight_hint, int):
        lines.append(f"preflight model_index_total={_fmt_bytes(weight_hint)}")
    return lines


def oom_backoff_recipe(recipe: FinetuneRecipe) -> FinetuneRecipe:
    from dataclasses import replace

    if recipe.per_device_train_batch_size <= 1:
        raise RuntimeError(OOM_HINT)
    scale = recipe.per_device_train_batch_size
    return replace(
        recipe,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps * scale,
    )
