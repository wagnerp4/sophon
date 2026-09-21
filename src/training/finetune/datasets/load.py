from __future__ import annotations

import csv
import json
from pathlib import Path

from training.finetune.datasets.formatters import format_alpaca_instruction_response
from training.finetune.datasets.registry import FINETUNE_DATASET_PRESETS, FinetuneDatasetPreset
from training.finetune.recipe import FinetuneRecipe
from utils.device.env_bootstrap import sophon_project_root


def _require_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "finetune dependencies missing. Run: uv sync --extra finetune"
        ) from exc
    return datasets


def _cap_rows(dataset: object, recipe: FinetuneRecipe, preset: FinetuneDatasetPreset) -> object:
    max_examples = recipe.max_examples if recipe.max_examples > 0 else preset.max_examples
    if max_examples > 0 and len(dataset) > max_examples:
        return dataset.select(range(max_examples))
    return dataset


def _eos_token(tokenizer: object | None) -> str:
    if tokenizer is None:
        return ""
    eos = getattr(tokenizer, "eos_token", None)
    if isinstance(eos, str):
        return eos
    return ""


def _to_dataset(rows: list[dict[str, str]]):
    _require_datasets()
    from datasets import Dataset

    return Dataset.from_list(rows)


def load_alpaca_hub_dataset(
    preset: FinetuneDatasetPreset,
    recipe: FinetuneRecipe,
    *,
    tokenizer: object | None = None,
    split: str | None = None,
) -> object:
    _require_datasets()
    from datasets import load_dataset

    if not preset.hub_id:
        raise ValueError("alpaca hub dataset requires hub_id")
    dataset = load_dataset(preset.hub_id, split=split or preset.split)
    dataset = _cap_rows(dataset, recipe, preset)
    eos_token = _eos_token(tokenizer)
    instruction_field = preset.instruction_field
    response_field = preset.response_field

    def formatting_prompts(examples: dict) -> dict[str, list[str]]:
        instructions = examples[instruction_field]
        responses = examples[response_field]
        texts: list[str] = []
        for instruction, response in zip(instructions, responses):
            texts.append(
                format_alpaca_instruction_response(
                    str(instruction),
                    str(response),
                    eos_token=eos_token,
                )
            )
        return {"text": texts}

    return dataset.map(formatting_prompts, batched=True)


def _hellaswag_jsonl_path(project_root: Path, split: str) -> Path:
    name = f"hellaswag_{split}.jsonl"
    candidates = (
        project_root / "data" / "hellaswag" / "data" / name,
        project_root / "data" / "benchmarks" / "hellaswag" / name,
        project_root / "data" / "hellaswag" / name,
    )
    for path in candidates:
        if path.is_file():
            return path
    searched = " | ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"hellaswag jsonl for split {split!r} not found. Looked in: {searched}"
    )


def _mmlu_split_aliases(split: str) -> tuple[str, ...]:
    if split == "train":
        return ("train", "auxiliary_train", "dev")
    if split in {"val", "validation"}:
        return ("val", "validation", "dev")
    return (split,)


def _iter_mmlu_csv_file(path: Path, subject: str) -> list[tuple[str, str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        has_header = False
        if sample:
            try:
                has_header = csv.Sniffer().has_header(sample)
            except csv.Error:
                has_header = False
        reader = csv.reader(fh)
        header: list[str] | None = None
        if has_header:
            header_row = next(reader, None)
            if header_row is not None:
                header = [cell.strip().lower() for cell in header_row]
        for row in reader:
            if len(row) < 6:
                continue
            if header is not None and {"question", "a", "b", "c", "d"} <= set(header):
                mapped = {header[i]: row[i] for i in range(min(len(header), len(row)))}
                question = mapped.get("question", row[0])
                a = mapped.get("a", row[1])
                b = mapped.get("b", row[2])
                c = mapped.get("c", row[3])
                d = mapped.get("d", row[4])
                label = mapped.get("label", mapped.get("answer", row[5]))
                subj = mapped.get("subject", subject)
            else:
                question, a, b, c, d, label = row[0], row[1], row[2], row[3], row[4], row[5]
                subj = subject
            rows.append((subj, question, a, b, c, d, str(label).strip()))
    return rows


def _mmlu_rows(project_root: Path, split: str) -> list[tuple[str, str, str, str, str, str, str]]:
    for alias in _mmlu_split_aliases(split):
        files = (
            project_root / "data" / "benchmarks" / "mmlu" / f"mmlu_{alias}.csv",
            project_root / "data" / "benchmarks" / "mmlu" / "data" / f"{alias}.csv",
            project_root / "data" / "test" / "data" / f"mmlu_{alias}.csv",
        )
        for path in files:
            if path.is_file():
                return _iter_mmlu_csv_file(path, "mmlu")
        dirs = (
            project_root / "data" / "test" / "data" / alias,
            project_root / "data" / "benchmarks" / "mmlu" / "data" / alias,
            project_root / "data" / "mmlu" / alias,
        )
        for directory in dirs:
            if not directory.is_dir():
                continue
            suffix = f"_{alias}.csv"
            csv_files = sorted(
                p for p in directory.iterdir() if p.is_file() and p.name.endswith(suffix)
            )
            if not csv_files:
                csv_files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".csv")
            collected: list[tuple[str, str, str, str, str, str, str]] = []
            for csv_path in csv_files:
                subject = csv_path.name[: -len(suffix)] if csv_path.name.endswith(suffix) else csv_path.stem
                subject = subject.replace("_", " ")
                collected.extend(_iter_mmlu_csv_file(csv_path, subject))
            if collected:
                return collected
    raise FileNotFoundError(
        f"mmlu CSV for split {split!r} not found under data/test/data, "
        "data/benchmarks/mmlu, or data/mmlu"
    )


def _choice_letter(label: object, n_choices: int = 4) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_choices]
    if isinstance(label, int):
        if 0 <= label < n_choices:
            return letters[label]
        raise ValueError(f"choice label out of range: {label}")
    text = str(label).strip().upper()
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < n_choices:
            return letters[idx]
    if text and text[0] in letters:
        return text[0]
    raise ValueError(f"cannot parse choice label {label!r}")


def dataset_source_summary(preset: FinetuneDatasetPreset, project_root: Path | None = None) -> str:
    if preset.hub_id:
        return f"hub={preset.hub_id} split={preset.split}"
    root = project_root or sophon_project_root()
    if preset.local_kind == "hellaswag_jsonl":
        try:
            path = _hellaswag_jsonl_path(root, preset.split)
            return f"local={path}"
        except FileNotFoundError:
            return "local=hellaswag jsonl (missing on disk)"
    if preset.local_kind == "mmlu_csv":
        return f"local=mmlu csv split={preset.split}"
    return "source=unknown"


def _load_hellaswag(
    preset: FinetuneDatasetPreset,
    recipe: FinetuneRecipe,
    *,
    tokenizer: object | None,
    project_root: Path,
    split: str | None,
) -> object:
    path = _hellaswag_jsonl_path(project_root, split or preset.split)
    eos_token = _eos_token(tokenizer)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            endings = raw.get("endings")
            if not isinstance(endings, list) or not endings:
                continue
            try:
                letter = _choice_letter(raw.get("label"), len(endings))
            except ValueError:
                continue
            idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(letter)
            if idx >= len(endings):
                continue
            activity = str(raw.get("activity_label") or "").strip()
            ctx = str(raw.get("ctx") or "").strip()
            gold = str(endings[idx])
            instruction = (
                f"Activity: {activity}\nContext: {ctx}\n\n"
                "Write the most plausible continuation."
            )
            rows.append(
                {
                    "text": format_alpaca_instruction_response(
                        instruction,
                        gold,
                        eos_token=eos_token,
                    )
                }
            )
    dataset = _to_dataset(rows)
    return _cap_rows(dataset, recipe, preset)


def _load_mmlu(
    preset: FinetuneDatasetPreset,
    recipe: FinetuneRecipe,
    *,
    tokenizer: object | None,
    project_root: Path,
    split: str | None,
) -> object:
    eos_token = _eos_token(tokenizer)
    rows: list[dict[str, str]] = []
    for subject, question, a, b, c, d, label in _mmlu_rows(project_root, split or preset.split):
        try:
            letter = _choice_letter(label, 4)
        except ValueError:
            continue
        choices = {"A": a, "B": b, "C": c, "D": d}
        answer_text = choices.get(letter, "")
        instruction = (
            f"The following is a multiple choice question about {subject}.\n\n"
            f"{question}\n"
            f"A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
            "Reply with the letter and the answer text."
        )
        response = f"{letter}. {answer_text}".strip()
        rows.append(
            {
                "text": format_alpaca_instruction_response(
                    instruction,
                    response,
                    eos_token=eos_token,
                )
            }
        )
    dataset = _to_dataset(rows)
    return _cap_rows(dataset, recipe, preset)


def load_registered_dataset(
    dataset_id: str,
    recipe: FinetuneRecipe,
    *,
    tokenizer: object | None = None,
    project_root: Path | None = None,
    split: str | None = None,
) -> object:
    preset = FINETUNE_DATASET_PRESETS.get(dataset_id)
    if preset is None:
        raise ValueError(f"unknown finetune dataset preset: {dataset_id!r}")
    root = project_root or sophon_project_root()
    if preset.formatter_id == "alpaca_instruction_response":
        return load_alpaca_hub_dataset(preset, recipe, tokenizer=tokenizer, split=split)
    if preset.local_kind == "hellaswag_jsonl":
        return _load_hellaswag(
            preset,
            recipe,
            tokenizer=tokenizer,
            project_root=root,
            split=split,
        )
    if preset.local_kind == "mmlu_csv":
        return _load_mmlu(
            preset,
            recipe,
            tokenizer=tokenizer,
            project_root=root,
            split=split,
        )
    raise ValueError(f"unsupported formatter_id: {preset.formatter_id!r}")


def maybe_split_eval(
    train_dataset: object,
    recipe: FinetuneRecipe,
    *,
    dataset_id: str,
    tokenizer: object | None = None,
    project_root: Path | None = None,
) -> tuple[object, object | None]:
    split = (recipe.eval_split or "").strip()
    if not split:
        return train_dataset, None
    try:
        fraction = float(split)
    except ValueError:
        fraction = None
    if fraction is not None:
        if fraction <= 0 or fraction >= 1:
            raise ValueError("eval_split as a fraction must be in (0, 1)")
        split_ds = train_dataset.train_test_split(test_size=fraction, seed=recipe.seed)
        return split_ds["train"], split_ds["test"]
    eval_dataset = load_registered_dataset(
        dataset_id,
        recipe,
        tokenizer=tokenizer,
        project_root=project_root,
        split=split,
    )
    return train_dataset, eval_dataset
