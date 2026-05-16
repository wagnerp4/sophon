from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HFModelPreset:
    repo_id: str
    default_local_dir: str
    description: str


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


HF_MODEL_PRESETS: dict[str, HFModelPreset] = {
    **_preset_dict(_GOOGLE_PRESET_ROWS),
    **_preset_dict(_META_PRESET_ROWS),
    **_preset_dict(_VICUNA_PRESET_ROWS),
    **_preset_dict(_QWEN_PRESET_ROWS),
}


def preset_keys_sorted() -> list[str]:
    return sorted(HF_MODEL_PRESETS.keys())


def resolve_preset_dir(preset_key: str, cwd: Path | None = None) -> Path:
    preset = HF_MODEL_PRESETS[preset_key]
    base = cwd if cwd is not None else Path.cwd()
    return (base / preset.default_local_dir).expanduser().resolve()


PREFERRED_DEFAULT_KEY = "llama2_7b_chat"


def default_preset_key() -> str:
    from utils.env_bootstrap import lmwrap_project_root

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
