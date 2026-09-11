from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HellaswagTransformConfig:
    ctx_mode: str
    lm_eval_normalize: bool
    split_types_allowlist: frozenset[str] | None
    shuffle_choices_seed: int | None


def hellaswag_config_from_mapping(raw: dict[str, Any]) -> HellaswagTransformConfig:
    ctx_mode = str(raw.get("ctx_mode") or "raw")
    if ctx_mode not in ("raw", "lm_eval"):
        raise ValueError(f"Hellaswag ctx_mode must be 'raw' or 'lm_eval', got {ctx_mode!r}")
    lm_eval_normalize = bool(raw.get("lm_eval_normalize"))
    split_types = raw.get("split_types")
    allowlist = None
    if split_types is not None:
        if not isinstance(split_types, list):
            raise ValueError("split_types must be a list of strings or null")
        allowlist = frozenset(str(x) for x in split_types)
    shuffle_raw = raw.get("shuffle_choices_seed")
    shuffle_seed = None
    if shuffle_raw is not None:
        if isinstance(shuffle_raw, bool) or not isinstance(shuffle_raw, int):
            raise ValueError("shuffle_choices_seed must be an integer or null")
        shuffle_seed = int(shuffle_raw)
    return HellaswagTransformConfig(
        ctx_mode=ctx_mode,
        lm_eval_normalize=lm_eval_normalize,
        split_types_allowlist=allowlist,
        shuffle_choices_seed=shuffle_seed,
    )
