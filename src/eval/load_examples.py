from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from eval.task_spec import BenchmarkTask


_MMLU_LETTERS = ("A", "B", "C", "D")


@dataclass
class Example:
    idx: int
    raw: dict[str, object]
    subject: str | None = None
    split: str | None = None
    fields: dict[str, object] = field(default_factory=dict)


def _open_jsonl(path: Path) -> Iterator[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"jsonl source not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {i} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {i} of {path} is not a JSON object")
            yield row


def _map_fields(row: dict[str, object], field_map: dict[str, str]) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for placeholder, source_key in field_map.items():
        if source_key in row:
            mapped[placeholder] = row[source_key]
    return mapped


def _iter_jsonl_examples(task: BenchmarkTask, limit: int | None) -> Iterator[Example]:
    assert task.source.path is not None
    for i, row in enumerate(_open_jsonl(task.source.path)):
        if limit is not None and i >= limit:
            break
        mapped = _map_fields(row, task.prompt.field_map)
        yield Example(idx=i, raw=row, fields=mapped)


def _iter_mmlu_examples(
    task: BenchmarkTask,
    split: str | None,
    limit: int | None,
) -> Iterator[Example]:
    assert task.source.root is not None
    splits = task.source.splits or {"test": "test"}
    requested = split or ("test" if "test" in splits else next(iter(splits.keys())))
    if requested not in splits:
        raise ValueError(
            f"mmlu split {requested!r} not declared in manifest splits {list(splits.keys())}"
        )
    split_dir = task.source.root / splits[requested]
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"MMLU split directory missing: {split_dir}. "
            "Download data.tar from the upstream MMLU repo and extract it under data/test/data."
        )
    suffix = f"_{requested}.csv"
    csv_files = sorted(p for p in split_dir.iterdir() if p.is_file() and p.name.endswith(suffix))
    if not csv_files:
        raise FileNotFoundError(
            f"no '*{suffix}' files in {split_dir}. "
            "Expected one CSV per subject from the upstream MMLU dump."
        )
    yielded = 0
    for csv_path in csv_files:
        subject = csv_path.name[: -len(suffix)].replace("_", " ")
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            for row_idx, row in enumerate(reader):
                if len(row) < 6:
                    continue
                question, a, b, c, d, label = row[0], row[1], row[2], row[3], row[4], row[5]
                raw = {
                    "subject": subject,
                    "question": question,
                    "a": a,
                    "b": b,
                    "c": c,
                    "d": d,
                    "label": label.strip().upper(),
                    "csv_row": row_idx,
                    "csv_file": csv_path.name,
                }
                mapped = _map_fields(raw, task.prompt.field_map)
                mapped.setdefault("subject", subject)
                mapped.setdefault("a", a)
                mapped.setdefault("b", b)
                mapped.setdefault("c", c)
                mapped.setdefault("d", d)
                ex = Example(
                    idx=yielded,
                    raw=raw,
                    subject=subject,
                    split=requested,
                    fields=mapped,
                )
                yield ex
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def iter_examples(
    task: BenchmarkTask,
    *,
    split: str | None = None,
    limit: int | None = None,
) -> Iterable[Example]:
    if task.source.kind == "jsonl":
        return _iter_jsonl_examples(task, limit)
    if task.source.kind == "mmlu_csv":
        return _iter_mmlu_examples(task, split, limit)
    if task.source.kind == "hellaswag":
        from eval.datasets.hellaswag.pipeline import iter_hellaswag_examples as iter_hs

        return iter_hs(task, split=split, limit=limit)
    raise ValueError(f"unsupported source kind: {task.source.kind!r}")


def render_choices_block(choices: list[str], num_choices: int) -> str:
    letters = _MMLU_LETTERS[:num_choices]
    rows: list[str] = []
    for letter, text in zip(letters, choices):
        rows.append(f"{letter}. {text}")
    return "\n".join(rows)


def extract_choices(example: Example, num_choices: int) -> list[str] | None:
    fields = example.fields
    raw = example.raw
    if "choices" in fields and isinstance(fields["choices"], list):
        choices = [str(x) for x in fields["choices"]]
        return choices[:num_choices]
    letter_keys = ("a", "b", "c", "d", "e", "f")
    collected: list[str] = []
    for key in letter_keys[:num_choices]:
        if key in fields:
            collected.append(str(fields[key]))
        elif key in raw:
            collected.append(str(raw[key]))
        else:
            return None
    return collected


def render_prompt(task: BenchmarkTask, example: Example) -> str:
    template = task.prompt.template
    fields = dict(example.fields)
    choices = extract_choices(example, task.scorer.num_choices)
    if choices is not None:
        fields["choices_block"] = render_choices_block(choices, task.scorer.num_choices)
    safe_fields: dict[str, object] = {k: ("" if v is None else v) for k, v in fields.items()}
    try:
        return template.format(**safe_fields)
    except KeyError as exc:
        missing = exc.args[0] if exc.args else "?"
        raise KeyError(
            f"prompt template for task {task.id!r} references "
            f"placeholder {missing!r} that is missing in example fields"
        ) from exc
