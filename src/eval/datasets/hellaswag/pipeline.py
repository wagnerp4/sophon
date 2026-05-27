from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from eval.datasets.hellaswag.io import iter_jsonl_objects
from eval.datasets.hellaswag.options import hellaswag_config_from_mapping
from eval.datasets.hellaswag.transform import transform_row_for_prompt
from eval.load_examples import Example
from eval.task_spec import BenchmarkTask


# TODO: Add optional likelihood-ranking scorer over ctx+ending pairs to match lm-evaluation-harness acc_norm style metrics.


def hellaswag_split_jsonl(task: BenchmarkTask, requested: str | None) -> Path:
    assert task.source.root is not None
    splits = task.source.splits
    if not splits:
        raise ValueError("hellaswag source.splits must be non-empty")
    split_key = requested or ("val" if "val" in splits else next(iter(splits.keys())))
    if split_key not in splits:
        raise ValueError(f"split {split_key!r} not in manifest splits {sorted(splits.keys())}")
    return (task.source.root / splits[split_key]).resolve()


def iter_hellaswag_examples(
    task: BenchmarkTask,
    *,
    split: str | None,
    limit: int | None,
) -> Iterator[Example]:
    cfg = hellaswag_config_from_mapping(dict(task.source.dataset_options))
    split_key = split or ("val" if "val" in task.source.splits else next(iter(task.source.splits.keys())))
    if split_key == "test":
        raise ValueError(
            "hellaswag public test JSONL omits labels; use split val or train for scored runs."
        )
    path = hellaswag_split_jsonl(task, split)
    yielded = 0
    num_choices = task.scorer.num_choices
    for i, row_any in enumerate(iter_jsonl_objects(path)):
        row = dict(row_any)
        if cfg.split_types_allowlist is not None:
            st = row.get("split_type")
            if str(st) not in cfg.split_types_allowlist:
                continue
        stable_key = int(row["ind"]) if isinstance(row.get("ind"), int) else i
        transformed = transform_row_for_prompt(
            row,
            cfg,
            stable_shuffle_key=stable_key,
            num_choices=num_choices,
        )
        mapped: dict[str, object] = {}
        for placeholder, source_key in task.prompt.field_map.items():
            if source_key in transformed:
                mapped[placeholder] = transformed[source_key]
        ex = Example(
            idx=yielded,
            raw=transformed,
            subject=str(transformed.get("activity_label")),
            split=split_key,
            fields=mapped,
        )
        yield ex
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def hellaswag_supported_splits(task: BenchmarkTask) -> list[str]:
    return sorted(task.source.splits.keys())
