from __future__ import annotations

import os
from pathlib import Path

DEFAULT_LOCAL_DIR = "models/meta-llama-Llama-2-7b-chat-hf"


def qbit_to_quantization(qbit: int) -> str:
    m = {0: "none", 4: "4bit", 8: "8bit"}
    if qbit not in m:
        raise ValueError(f"qbit must be 0, 4, or 8, got {qbit!r}")
    return m[qbit]


def resolve_quantization_choice(value: str) -> str:
    v = value.strip()
    if v == "lightweight":
        return "4bit"
    return v


def resolve_cli_quantization(*, qbit: int | None, quantization: str) -> str:
    if qbit is not None:
        return qbit_to_quantization(qbit)
    return resolve_quantization_choice(quantization)


def resolve_local_model_dir(model_arg: str | None) -> Path:
    if model_arg:
        return Path(model_arg).expanduser().resolve()
    env_path = os.environ.get("GEMMA4_MODEL")
    if env_path:
        return Path(env_path).expanduser().resolve()
    base = os.environ.get("GEMMA4_LOCAL_DIR", DEFAULT_LOCAL_DIR)
    return Path(base).expanduser().resolve()


def require_model_on_disk(model_dir: Path) -> str:
    if not (model_dir / "config.json").is_file():
        raise ValueError(
            "Expected a local model directory on disk with config.json at "
            f"{model_dir}. Set --model, GEMMA4_MODEL, or GEMMA4_LOCAL_DIR."
        )
    return str(model_dir)


def infer_default_quantization() -> str:
    qbit_raw = os.environ.get("GEMMA4_QBIT", "").strip()
    if qbit_raw in ("0", "4", "8"):
        return qbit_to_quantization(int(qbit_raw))
    raw = os.environ.get("GEMMA4_QUANTIZATION", "").strip()
    if raw:
        resolved = resolve_quantization_choice(raw)
        if resolved in ("none", "4bit", "8bit"):
            return resolved
    try:
        import torch

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "none"
    except Exception:
        pass
    return "8bit"
