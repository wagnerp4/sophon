from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def slugify_segment(raw: str, max_len: int = 56) -> str:
    s = raw.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_.+-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "unknown"


def sanitize_run_id_component(raw: str, max_len: int = 200) -> str:
    s = raw.strip().replace("\\", "_").replace("/", "_").replace("..", "_")
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "run"


@dataclass(frozen=True)
class BenchFingerprintParts:
    backend: str
    model_slug: str
    quantization: str
    enable_thinking: bool
    ollama_model: str | None
    ollama_base_url: str | None
    max_new_tokens_override: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    seed: int | None
    split: str | None
    limit: int | None


def fingerprint_payload(parts: BenchFingerprintParts) -> dict[str, Any]:
    return {
        "backend": parts.backend,
        "model_slug": parts.model_slug,
        "quantization": parts.quantization,
        "enable_thinking": parts.enable_thinking,
        "ollama_model": parts.ollama_model,
        "ollama_base_url": parts.ollama_base_url,
        "cli_max_new_tokens": parts.max_new_tokens_override,
        "cli_temperature": parts.temperature,
        "cli_top_p": parts.top_p,
        "cli_top_k": parts.top_k,
        "cli_repetition_penalty": parts.repetition_penalty,
        "cli_seed": parts.seed,
        "cli_split": parts.split,
        "cli_limit": parts.limit,
    }


def fingerprint_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def default_run_id(parts: BenchFingerprintParts, *, ts: datetime | None = None) -> str:
    stamp = ts or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    else:
        stamp = stamp.astimezone(timezone.utc)
    ts_compact = f"{stamp.strftime('%Y%m%dT%H%M%S')}_{stamp.microsecond:06d}Z"
    fp = fingerprint_payload(parts)
    digest = fingerprint_hash(fp)
    model = slugify_segment(parts.model_slug, max_len=48)
    backend = slugify_segment(parts.backend, max_len=16)
    quant = slugify_segment(parts.quantization, max_len=12)
    think = "think1" if parts.enable_thinking else "think0"
    return sanitize_run_id_component(f"{ts_compact}_{backend}_{model}_{quant}_{think}_{digest}")
