from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

_WEIGHT_SHARD_RE = re.compile(r"^(model|pytorch_model)-(\d+)-of-(\d+)\.(safetensors|bin)$")


@dataclass(frozen=True)
class HFModelPreset:
    repo_id: str
    default_local_dir: str
    description: str
    finetune_eligible: bool = False
    finetune_requires_allow_large: bool = False
    finetune_reason: str = ""
    vram_class: str = ""


PresetRow = tuple[str, str, str] | tuple[str, str, str, str]


def _preset_dict(rows: tuple[PresetRow, ...]) -> dict[str, HFModelPreset]:
    out: dict[str, HFModelPreset] = {}
    for row in rows:
        if len(row) == 4:
            key, repo_id, desc, local_rel = row
            default_local_dir = local_rel
        else:
            key, repo_id, desc = row
            default_local_dir = f"models/{repo_id.replace('/', '-')}"
        out[key] = HFModelPreset(
            repo_id=repo_id,
            default_local_dir=default_local_dir,
            description=desc,
        )
    return out


_GOOGLE_PRESET_ROWS: tuple[PresetRow, ...] = (
    (  # Hub: image-text-to-text
        "gemma4_31b_it",
        "google/gemma-4-31B-it",
        "Gemma 4 31B image-text-to-text instruct.",
        "models/google-gemma-4-31b-it",
    ),
    ("gemma4_31b", "google/gemma-4-31B", "Gemma 4 31B image-text-to-text base."),  # Hub: image-text-to-text
    ("gemma4_26b_a4b_it", "google/gemma-4-26B-A4B-it", "Gemma 4 26B-A4B MoE image-text-to-text instruct."),  # Hub: image-text-to-text
    ("gemma4_26b_a4b", "google/gemma-4-26B-A4B", "Gemma 4 26B-A4B MoE image-text-to-text base."),  # Hub: image-text-to-text
    (  # Hub: any-to-any
        "gemma4_12b_it",
        "google/gemma-4-12B-it",
        "Gemma 4 12B unified encoder-free any-to-any instruct.",
        "models/google-gemma-4-12b-it",
    ),
    ("gemma4_e4b_it", "google/gemma-4-E4B-it", "Gemma 4 E4B (~8B) any-to-any instruct."),  # Hub: any-to-any
    ("gemma4_e4b", "google/gemma-4-E4B", "Gemma 4 E4B (~8B) any-to-any base."),  # Hub: any-to-any
    ("gemma4_e2b_it", "google/gemma-4-E2B-it", "Gemma 4 E2B (~5B) any-to-any instruct."),  # Hub: any-to-any
    ("gemma4_e2b", "google/gemma-4-E2B", "Gemma 4 E2B (~5B) any-to-any base."),  # Hub: any-to-any
    ("gemma4_e2b_it_assistant", "google/gemma-4-E2B-it-assistant", "Gemma 4 E2B instruct assistant variant."),  # Hub: any-to-any
    ("gemma4_e4b_it_assistant", "google/gemma-4-E4B-it-assistant", "Gemma 4 E4B instruct assistant variant."),  # Hub: any-to-any
    (  # Hub: any-to-any
        "gemma4_26b_a4b_it_assistant",
        "google/gemma-4-26B-A4B-it-assistant",
        "Gemma 4 26B-A4B instruct assistant variant.",
    ),
    ("gemma4_31b_it_assistant", "google/gemma-4-31B-it-assistant", "Gemma 4 31B instruct assistant variant."),  # Hub: any-to-any
)


_META_PRESET_ROWS: tuple[PresetRow, ...] = (
    ("llama2_7b_chat", "meta-llama/Llama-2-7b-chat-hf", "Llama 2 7B chat HF weights (gated on the Hub)."),  # Hub: text-generation
    ("llama2_13b_chat", "meta-llama/Llama-2-13b-chat-hf", "Llama 2 13B chat HF weights (gated on the Hub)."),  # Hub: text-generation
    ("llama3_8b_instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "Llama 3 8B instruct (gated on the Hub)."),  # Hub: text-generation
    ("llama3_70b_instruct", "meta-llama/Meta-Llama-3-70B-Instruct", "Llama 3 70B instruct (gated on the Hub)."),  # Hub: text-generation
    ("llama3_1_8b_instruct", "meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B instruct (gated; canonical Hub id)."),  # Hub: text-generation
    ("llama4_scout_17b", "meta-llama/Llama-4-Scout-17B-16E", "Llama 4 Scout MoE (gated on the Hub)."),  # Hub: image-text-to-text
    (  # Hub: image-text-to-text
        "llama4_maverick_17b_instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-Original",
        "Llama 4 Maverick instruct MoE (gated on the Hub).",
    ),
)


_VICUNA_PRESET_ROWS: tuple[PresetRow, ...] = (
    ("vicuna_7b_v1_5", "lmsys/vicuna-7b-v1.5", "Vicuna 7B v1.5 (Llama 2-derived chat fine-tune)."),  # Hub: text-generation
    ("vicuna_13b_v1_5", "lmsys/vicuna-13b-v1.5", "Vicuna 13B v1.5 (Llama 2-derived chat fine-tune)."),  # Hub: text-generation
)


_QWEN_PRESET_ROWS: tuple[PresetRow, ...] = (
    ("qwen3_6_35b_a3b", "Qwen/Qwen3.6-35B-A3B", "Qwen3.6 35B-A3B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_6_35b_a3b_fp8", "Qwen/Qwen3.6-35B-A3B-FP8", "Qwen3.6 35B-A3B FP8 image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_6_27b", "Qwen/Qwen3.6-27B", "Qwen3.6 27B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_6_27b_fp8", "Qwen/Qwen3.6-27B-FP8", "Qwen3.6 27B FP8 image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_tts_12hz_17b_customvoice", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3 TTS 1.7B custom voice."),  # Hub: text-to-speech
    ("qwen3_tts_12hz_06b_base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen3 TTS 0.6B base."),  # Hub: text-to-speech
    ("qwen3_tts_12hz_17b_voicedesign", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen3 TTS 1.7B voice design."),  # Hub: text-to-speech
    ("qwen3_tts_12hz_17b_base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen3 TTS 1.7B base."),  # Hub: text-to-speech
    ("qwen3_tts_12hz_06b_customvoice", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen3 TTS 0.6B custom voice."),  # Hub: text-to-speech
    ("qwen3_tts_tokenizer_12hz", "Qwen/Qwen3-TTS-Tokenizer-12Hz", "Qwen3 TTS tokenizer (12 Hz)."),  # Hub: audio-to-audio
    ("qwen3_asr_17b", "Qwen/Qwen3-ASR-1.7B", "Qwen3 ASR 1.7B."),  # Hub: automatic-speech-recognition
    ("qwen3_asr_06b", "Qwen/Qwen3-ASR-0.6B", "Qwen3 ASR 0.6B."),  # Hub: automatic-speech-recognition
    ("qwen3_forced_aligner_06b", "Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3 forced aligner 0.6B."),  # Hub: automatic-speech-recognition
    ("qwen3_5_397b_a17b", "Qwen/Qwen3.5-397B-A17B", "Qwen3.5 397B-A17B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_397b_a17b_fp8", "Qwen/Qwen3.5-397B-A17B-FP8", "Qwen3.5 397B-A17B FP8 image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_122b_a10b", "Qwen/Qwen3.5-122B-A10B", "Qwen3.5 122B-A10B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_122b_a10b_fp8", "Qwen/Qwen3.5-122B-A10B-FP8", "Qwen3.5 122B-A10B FP8 image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_35b_a3b", "Qwen/Qwen3.5-35B-A3B", "Qwen3.5 35B-A3B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_35b_a3b_fp8", "Qwen/Qwen3.5-35B-A3B-FP8", "Qwen3.5 35B-A3B FP8 image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_35b_a3b_base", "Qwen/Qwen3.5-35B-A3B-Base", "Qwen3.5 35B-A3B base image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_27b", "Qwen/Qwen3.5-27B", "Qwen3.5 27B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_27b_fp8", "Qwen/Qwen3.5-27B-FP8", "Qwen3.5 27B FP8 image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_9b", "Qwen/Qwen3.5-9B", "Qwen3.5 9B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_9b_base", "Qwen/Qwen3.5-9B-Base", "Qwen3.5 9B base image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_4b", "Qwen/Qwen3.5-4B", "Qwen3.5 4B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_4b_base", "Qwen/Qwen3.5-4B-Base", "Qwen3.5 4B base image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_2b", "Qwen/Qwen3.5-2B", "Qwen3.5 2B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_2b_base", "Qwen/Qwen3.5-2B-Base", "Qwen3.5 2B base image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_0_8b", "Qwen/Qwen3.5-0.8B", "Qwen3.5 0.8B image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_0_8b_base", "Qwen/Qwen3.5-0.8B-Base", "Qwen3.5 0.8B base image-text-to-text."),  # Hub: image-text-to-text
    ("qwen3_5_397b_a17b_gptq_int4", "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4", "Qwen3.5 397B-A17B GPTQ Int4."),  # Hub: image-text-to-text
    ("qwen3_5_122b_a10b_gptq_int4", "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4", "Qwen3.5 122B-A10B GPTQ Int4."),  # Hub: image-text-to-text
    ("qwen3_5_35b_a3b_gptq_int4", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", "Qwen3.5 35B-A3B GPTQ Int4."),  # Hub: image-text-to-text
    ("qwen3_5_27b_gptq_int4", "Qwen/Qwen3.5-27B-GPTQ-Int4", "Qwen3.5 27B GPTQ Int4."),  # Hub: image-text-to-text
)


_FINETUNE_STANDARD_VRAM: dict[str, str] = {
    "llama2_7b_chat": "7b",
    "llama2_13b_chat": "13b",
    "llama3_8b_instruct": "8b",
    "llama3_1_8b_instruct": "8b",
    "vicuna_7b_v1_5": "7b",
    "vicuna_13b_v1_5": "13b",
}

_FINETUNE_LARGE_VRAM: dict[str, str] = {
    "llama3_70b_instruct": "70b",
}


def _infer_finetune_ineligible_reason(key: str, preset: HFModelPreset) -> str:
    blob = f"{key} {preset.repo_id} {preset.description}".lower()
    if key.startswith("gemma4") or "gemma-4" in blob:
        return "Gemma 4 uses multimodal / image-text loaders, not causal LoRA SFT"
    if key.startswith("llama4") or "llama-4" in blob:
        return "Llama 4 is MoE vision, not causal LoRA SFT in this trainer"
    if "gptq" in blob:
        return "GPTQ checkpoints are already quantized; this trainer does not stack 4-bit LoRA on GPTQ"
    if "fp8" in blob:
        return "FP8 checkpoints are already quantized; this trainer does not stack 4-bit LoRA on FP8"
    if "tts" in blob:
        return "TTS weights are not a causal LoRA SFT target"
    if "asr" in blob or "forcedaligner" in blob or "tokenizer-12hz" in blob:
        return "ASR / audio weights are not a causal LoRA SFT target"
    if key.startswith("qwen"):
        return "Qwen rows in this registry are image-text, TTS, ASR, GPTQ, or FP8, not causal LoRA SFT"
    return "not a causal LoRA SFT target in this trainer"


def _with_finetune_flags(presets: dict[str, HFModelPreset]) -> dict[str, HFModelPreset]:
    out: dict[str, HFModelPreset] = {}
    for key, preset in presets.items():
        if key in _FINETUNE_STANDARD_VRAM:
            out[key] = replace(
                preset,
                finetune_eligible=True,
                finetune_requires_allow_large=False,
                finetune_reason="",
                vram_class=_FINETUNE_STANDARD_VRAM[key],
            )
            continue
        if key in _FINETUNE_LARGE_VRAM:
            out[key] = replace(
                preset,
                finetune_eligible=True,
                finetune_requires_allow_large=True,
                finetune_reason="requires allow_large=true (70B-class VRAM)",
                vram_class=_FINETUNE_LARGE_VRAM[key],
            )
            continue
        out[key] = replace(
            preset,
            finetune_eligible=False,
            finetune_requires_allow_large=False,
            finetune_reason=_infer_finetune_ineligible_reason(key, preset),
            vram_class="",
        )
    return out


HF_MODEL_PRESETS: dict[str, HFModelPreset] = _with_finetune_flags(
    {
        **_preset_dict(_GOOGLE_PRESET_ROWS),
        **_preset_dict(_META_PRESET_ROWS),
        **_preset_dict(_VICUNA_PRESET_ROWS),
        **_preset_dict(_QWEN_PRESET_ROWS),
    }
)


def preset_keys_sorted() -> list[str]:
    return sorted(HF_MODEL_PRESETS.keys())


def finetune_eligible_keys(*, include_large: bool = True) -> list[str]:
    keys: list[str] = []
    for key in preset_keys_sorted():
        preset = HF_MODEL_PRESETS[key]
        if not preset.finetune_eligible:
            continue
        if preset.finetune_requires_allow_large and not include_large:
            continue
        keys.append(key)
    return keys


def require_finetune_preset(preset_key: str, *, allow_large: bool = False) -> HFModelPreset:
    if preset_key not in HF_MODEL_PRESETS:
        raise ValueError(f"unknown HF preset {preset_key!r}")
    preset = HF_MODEL_PRESETS[preset_key]
    if not preset.finetune_eligible:
        reason = preset.finetune_reason or "not a causal LoRA SFT target in this trainer"
        raise ValueError(f"preset {preset_key!r} cannot be finetuned: {reason}")
    if preset.finetune_requires_allow_large and not allow_large:
        raise ValueError(
            f"preset {preset_key!r} is 70B-class. Pass allow_large=true if VRAM is enough. "
            f"{preset.finetune_reason}".strip()
        )
    return preset


def resolve_preset_dir(preset_key: str, cwd: Path | None = None) -> Path:
    preset = HF_MODEL_PRESETS[preset_key]
    base = cwd if cwd is not None else Path.cwd()
    return (base / preset.default_local_dir).expanduser().resolve()


def _index_shards_complete(model_dir: Path, index_name: str) -> bool:
    index_path = model_dir / index_name
    if not index_path.is_file():
        return False
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards = {str(name) for name in weight_map.values()}
    return bool(shards) and all((model_dir / shard).is_file() for shard in shards)


def _numbered_shards_complete(model_dir: Path, prefix: str, suffix: str) -> bool:
    shards_by_total: dict[int, set[int]] = {}
    for path in model_dir.iterdir():
        if not path.is_file():
            continue
        match = _WEIGHT_SHARD_RE.match(path.name)
        if match is None or match.group(1) != prefix or match.group(4) != suffix:
            continue
        shard_index = int(match.group(2))
        shard_total = int(match.group(3))
        shards_by_total.setdefault(shard_total, set()).add(shard_index)
    for shard_total, found in shards_by_total.items():
        if found == set(range(1, shard_total + 1)):
            return True
    return False


def model_dir_has_complete_weights(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    for single_name in ("model.safetensors", "pytorch_model.bin"):
        if (model_dir / single_name).is_file():
            return True
    if _index_shards_complete(model_dir, "model.safetensors.index.json"):
        return True
    if _index_shards_complete(model_dir, "pytorch_model.bin.index.json"):
        return True
    if _numbered_shards_complete(model_dir, "model", "safetensors"):
        return True
    if _numbered_shards_complete(model_dir, "pytorch_model", "bin"):
        return True
    return False


def preset_has_weights(preset_key: str, cwd: Path | None = None) -> bool:
    model_dir = resolve_preset_dir(preset_key, cwd)
    if not (model_dir / "config.json").is_file():
        return False
    return model_dir_has_complete_weights(model_dir)


def local_only_model_dirs(cwd: Path | None = None) -> list[Path]:
    from utils.device.env_bootstrap import sophon_project_root

    base = cwd if cwd is not None else sophon_project_root()
    models_root = base / "models"
    if not models_root.is_dir():
        return []
    preset_paths = {resolve_preset_dir(key, base).resolve() for key in HF_MODEL_PRESETS}
    out: list[Path] = []
    for sub in sorted(models_root.iterdir(), key=lambda item: item.name.lower()):
        if not sub.is_dir():
            continue
        resolved = sub.resolve()
        if (
            (resolved / "config.json").is_file()
            and model_dir_has_complete_weights(resolved)
            and resolved not in preset_paths
        ):
            out.append(resolved)
    return out


def match_preset_key(token: str) -> str | None:
    raw = token.strip()
    if not raw:
        return None
    if raw in HF_MODEL_PRESETS:
        return raw
    lower = raw.lower()
    exact_ci = [key for key in HF_MODEL_PRESETS if key.lower() == lower]
    if len(exact_ci) == 1:
        return exact_ci[0]
    prefix = [key for key in HF_MODEL_PRESETS if key.lower().startswith(lower)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        sample = ", ".join(prefix[:8])
        more = f" (+{len(prefix) - 8} more)" if len(prefix) > 8 else ""
        raise ValueError(f"ambiguous preset {raw!r}: {sample}{more}")
    return None


PREFERRED_DEFAULT_KEY = "llama2_7b_chat"


def preset_key_for_dir(model_dir: Path, cwd: Path | None = None) -> str | None:
    from utils.device.env_bootstrap import sophon_project_root

    base = cwd if cwd is not None else sophon_project_root()
    want = model_dir.expanduser().resolve()
    for key in preset_keys_sorted():
        if resolve_preset_dir(key, base).resolve() == want:
            return key
    return None


def default_local_preset_key(cwd: Path | None = None) -> str | None:
    from utils.device.env_bootstrap import sophon_project_root

    base = cwd if cwd is not None else sophon_project_root()
    if preset_has_weights(PREFERRED_DEFAULT_KEY, base):
        return PREFERRED_DEFAULT_KEY
    for key in preset_keys_sorted():
        if key == PREFERRED_DEFAULT_KEY:
            continue
        if preset_has_weights(key, base):
            return key
    return None


def resolve_chat_startup_model(
    *,
    preset: str | None,
    model: str | None,
    cwd: Path | None = None,
) -> tuple[str | None, Path]:
    from utils.device.env_bootstrap import sophon_project_root

    base = cwd if cwd is not None else sophon_project_root()
    fallback_dir = resolve_preset_dir(PREFERRED_DEFAULT_KEY, base)

    if preset is not None:
        return preset, resolve_preset_dir(preset, base)
    if model is not None:
        path = Path(model).expanduser().resolve()
        return preset_key_for_dir(path, base), path

    env_preset = os.environ.get("SOPHON_HF_PRESET", "").strip()
    if env_preset in HF_MODEL_PRESETS and preset_has_weights(env_preset, base):
        return env_preset, resolve_preset_dir(env_preset, base)

    env_path = os.environ.get("GEMMA4_MODEL", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if model_dir_has_complete_weights(path) and (path / "config.json").is_file():
            return preset_key_for_dir(path, base), path

    env_dir = os.environ.get("GEMMA4_LOCAL_DIR", "").strip()
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if model_dir_has_complete_weights(path) and (path / "config.json").is_file():
            return preset_key_for_dir(path, base), path

    local_key = default_local_preset_key(base)
    if local_key is not None:
        return local_key, resolve_preset_dir(local_key, base)

    for path in local_only_model_dirs(base):
        return None, path

    return None, fallback_dir


def default_preset_key() -> str:
    from utils.device.env_bootstrap import sophon_project_root

    base = sophon_project_root()
    raw = os.environ.get("SOPHON_HF_PRESET", "").strip()
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
    local_key = default_local_preset_key(base)
    if local_key is not None:
        return local_key
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
