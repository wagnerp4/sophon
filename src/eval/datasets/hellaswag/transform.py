from __future__ import annotations

import random
from typing import Any

from eval.datasets.hellaswag.options import HellaswagTransformConfig
from eval.datasets.hellaswag.text_clean import preprocess_lm_eval


def _capitalize_first_segment(ctx_b: str) -> str:
    ctx_b = ctx_b.strip()
    if not ctx_b:
        return ctx_b
    return ctx_b[0].upper() + ctx_b[1:]


def _normalize_maybe(text: str, cfg: HellaswagTransformConfig) -> str:
    if cfg.lm_eval_normalize:
        return preprocess_lm_eval(text)
    return text


def build_activity_ctx(row: dict[str, Any], cfg: HellaswagTransformConfig) -> tuple[str, str]:
    activity_raw = str(row.get("activity_label") or "")
    if cfg.ctx_mode == "lm_eval":
        ctx_a = str(row.get("ctx_a") or "")
        ctx_b = str(row.get("ctx_b") or "")
        ctx_raw = ctx_a.strip()
        seg_b = _capitalize_first_segment(ctx_b)
        if ctx_raw and seg_b:
            ctx_combined = ctx_raw + " " + seg_b
        elif ctx_raw:
            ctx_combined = ctx_raw
        else:
            ctx_combined = seg_b
    else:
        ctx_combined = str(row.get("ctx") or "")
    activity = _normalize_maybe(activity_raw, cfg)
    ctx_text = _normalize_maybe(ctx_combined, cfg)
    return activity, ctx_text


def normalize_endings(endings: list[str], cfg: HellaswagTransformConfig) -> list[str]:
    out = list(endings)
    if cfg.lm_eval_normalize:
        return [preprocess_lm_eval(str(e)) for e in out]
    return [str(e) for e in out]


def shuffle_endings(endings: list[str], label: int, rng: random.Random) -> tuple[list[str], int]:
    perm = list(range(len(endings)))
    rng.shuffle(perm)
    shuffled = [endings[i] for i in perm]
    new_label = perm.index(label)
    return shuffled, new_label


def transform_row_for_prompt(
    row: dict[str, Any],
    cfg: HellaswagTransformConfig,
    *,
    stable_shuffle_key: int,
    num_choices: int,
) -> dict[str, object]:
    endings_any = row.get("endings")
    if not isinstance(endings_any, list):
        raise ValueError("hellaswag row missing list field endings")
    endings = [str(x) for x in endings_any]
    if len(endings) != num_choices:
        raise ValueError(f"expected {num_choices} endings, got {len(endings)}")
    label_any = row.get("label")
    if label_any is None:
        raise ValueError("hellaswag row missing label (public test split has no gold answers)")
    label = int(label_any)
    if label < 0 or label >= num_choices:
        raise ValueError(f"label out of range: {label}")
    activity, ctx_text = build_activity_ctx(row, cfg)
    endings_norm = normalize_endings(endings, cfg)
    rng = None
    if cfg.shuffle_choices_seed is not None:
        rng = random.Random(int(cfg.shuffle_choices_seed) ^ (stable_shuffle_key & 0xFFFFFFFF))
    if rng is not None:
        endings_norm, label = shuffle_endings(endings_norm, label, rng)
    out: dict[str, object] = dict(row)
    out["activity_label"] = activity
    out["ctx"] = ctx_text
    out["endings"] = endings_norm
    out["label"] = label
    return out
