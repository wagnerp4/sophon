from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HFModelPreset:
    repo_id: str
    default_local_dir: str
    description: str


HF_MODEL_PRESETS: dict[str, HFModelPreset] = {
    "gemma4_31b_it": HFModelPreset(
        repo_id="google/gemma-4-31B-it",
        default_local_dir="models/google-gemma-4-31b-it",
        description="Gemma 4 31B instruct weights.",
    ),
    "gemma4_e2b_it": HFModelPreset(
        repo_id="google/gemma-4-E2B-it",
        default_local_dir="models/google-gemma-4-E2B-it",
        description="Gemma 4 E2B (~5B) instruct multimodal weights.",
    ),
    "gemma4_e4b_it": HFModelPreset(
        repo_id="google/gemma-4-E4B-it",
        default_local_dir="models/google-gemma-4-E4B-it",
        description="Gemma 4 E4B (~8B) instruct multimodal weights.",
    ),
    "gemma4_26b_a4b_it": HFModelPreset(
        repo_id="google/gemma-4-26B-A4B-it",
        default_local_dir="models/google-gemma-4-26B-A4B-it",
        description="Gemma 4 26B-A4B MoE instruct weights.",
    ),
    "llama2_7b_chat": HFModelPreset(
        repo_id="meta-llama/Llama-2-7b-chat-hf",
        default_local_dir="models/meta-llama-Llama-2-7b-chat-hf",
        description="Llama 2 7B chat HF weights (gated on the Hub).",
    ),
    "llama2_13b_chat": HFModelPreset(
        repo_id="meta-llama/Llama-2-13b-chat-hf",
        default_local_dir="models/meta-llama-Llama-2-13b-chat-hf",
        description="Llama 2 13B chat HF weights (gated on the Hub).",
    ),
    "llama3_8b_instruct": HFModelPreset(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        default_local_dir="models/meta-llama-Meta-Llama-3-8B-Instruct",
        description="Llama 3 8B instruct (gated on the Hub).",
    ),
    "llama3_70b_instruct": HFModelPreset(
        repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
        default_local_dir="models/meta-llama-Meta-Llama-3-70B-Instruct",
        description="Llama 3 70B instruct (gated on the Hub).",
    ),
    "llama3_1_8b_instruct": HFModelPreset(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        default_local_dir="models/meta-llama-Meta-Llama-3.1-8B-Instruct",
        description="Llama 3.1 8B instruct (gated; canonical Hub id).",
    ),
    "llama4_scout_17b": HFModelPreset(
        repo_id="meta-llama/Llama-4-Scout-17B-16E",
        default_local_dir="models/meta-llama-Llama-4-Scout-17B-16E",
        description="Llama 4 Scout MoE (gated on the Hub).",
    ),
    "llama4_maverick_17b_instruct": HFModelPreset(
        repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-Original",
        default_local_dir="models/meta-llama-Llama-4-Maverick-17B-128E-Instruct-Original",
        description="Llama 4 Maverick instruct MoE (gated on the Hub).",
    ),
    "vicuna_7b_v1_5": HFModelPreset(
        repo_id="lmsys/vicuna-7b-v1.5",
        default_local_dir="models/lmsys-vicuna-7b-v1.5",
        description="Vicuna 7B v1.5 (Llama 2-derived chat fine-tune).",
    ),
    "vicuna_13b_v1_5": HFModelPreset(
        repo_id="lmsys/vicuna-13b-v1.5",
        default_local_dir="models/lmsys-vicuna-13b-v1.5",
        description="Vicuna 13B v1.5 (Llama 2-derived chat fine-tune).",
    ),
}


def preset_keys_sorted() -> list[str]:
    return sorted(HF_MODEL_PRESETS.keys())


def resolve_preset_dir(preset_key: str, cwd: Path | None = None) -> Path:
    preset = HF_MODEL_PRESETS[preset_key]
    base = cwd if cwd is not None else Path.cwd()
    return (base / preset.default_local_dir).expanduser().resolve()


PREFERRED_DEFAULT_KEY = "llama2_7b_chat"


def default_preset_key() -> str:
    from lmwrap.utils.env_bootstrap import lmwrap_project_root

    base = lmwrap_project_root()
    raw = os.environ.get("LMWRAP_HF_PRESET", "").strip()
    if raw in HF_MODEL_PRESETS:
        return raw
    env_path = os.environ.get("GEMMA4_MODEL", "").strip()
    if env_path:
        want = Path(env_path).expanduser().resolve()
        for key in preset_keys_sorted():
            if resolve_preset_dir(key, base) == want:
                return key
    env_dir = os.environ.get("GEMMA4_LOCAL_DIR", "").strip()
    if env_dir:
        want = Path(env_dir).expanduser().resolve()
        for key in preset_keys_sorted():
            pk = resolve_preset_dir(key, base)
            if pk == want:
                return key
            if pk.name == want.name:
                return key
    if PREFERRED_DEFAULT_KEY in HF_MODEL_PRESETS:
        p_pref = resolve_preset_dir(PREFERRED_DEFAULT_KEY, base)
        if (p_pref / "config.json").is_file():
            return PREFERRED_DEFAULT_KEY
    for key in preset_keys_sorted():
        p = resolve_preset_dir(key, base)
        if (p / "config.json").is_file():
            return key
    models_root = base / "models"
    if models_root.is_dir():
        for sub in sorted(models_root.iterdir(), key=lambda x: x.name.lower()):
            if not sub.is_dir():
                continue
            sr = sub.resolve()
            for key in preset_keys_sorted():
                if resolve_preset_dir(key, base).resolve() == sr:
                    return key
    return PREFERRED_DEFAULT_KEY


def preset_summary_lines() -> list[str]:
    lines: list[str] = []
    for key in sorted(HF_MODEL_PRESETS.keys()):
        p = HF_MODEL_PRESETS[key]
        lines.append(f"{key}: {p.repo_id}")
        if p.description:
            lines.append(f"    {p.description}")
    return lines
