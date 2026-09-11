from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass

from eval.load_examples import Example, extract_choices
from eval.task_spec import BenchmarkTask, ScorerSpec


@dataclass
class ScoreResult:
    correct: bool
    predicted: str | None
    gold: str
    notes: str = ""


_LETTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:answer|final\s*answer|the\s*answer)\s*(?:is|:)?\s*\(?([A-Z])\)?", re.IGNORECASE),
    re.compile(r"\(([A-Z])\)"),
    re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])"),
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _letters_for(num_choices: int) -> tuple[str, ...]:
    return tuple(chr(ord("A") + i) for i in range(num_choices))


def _label_to_letter(label: object, num_choices: int) -> str:
    letters = _letters_for(num_choices)
    if isinstance(label, bool):
        raise ValueError(f"label {label!r} is boolean, expected int or letter")
    if isinstance(label, int):
        if 0 <= label < num_choices:
            return letters[label]
        raise ValueError(f"label {label} out of range for {num_choices} choices")
    if isinstance(label, str):
        s = label.strip().upper()
        if len(s) == 1 and s in letters:
            return s
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < num_choices:
                return letters[idx]
        raise ValueError(f"cannot interpret label {label!r} as one of {letters}")
    raise ValueError(f"unsupported label type {type(label).__name__}")


def _extract_letter(text: str, num_choices: int) -> str | None:
    letters = set(_letters_for(num_choices))
    if not text:
        return None
    upper = text.strip()
    for pattern in _LETTER_PATTERNS:
        for m in pattern.finditer(upper):
            ch = m.group(1).upper()
            if ch in letters:
                return ch
    return None


def _extract_by_choice_overlap(text: str, choices: list[str], num_choices: int) -> str | None:
    letters = _letters_for(num_choices)
    norm_pred = _normalize(text)
    if not norm_pred:
        return None
    best: tuple[int, str] | None = None
    for letter, choice in zip(letters, choices):
        norm_choice = _normalize(choice)
        if not norm_choice:
            continue
        pos = norm_pred.find(norm_choice)
        if pos < 0:
            continue
        if best is None or pos < best[0]:
            best = (pos, letter)
    return best[1] if best is not None else None


def _score_multiple_choice_letter(
    prediction: str,
    example: Example,
    task: BenchmarkTask,
    scorer: ScorerSpec,
) -> ScoreResult:
    label_raw = example.fields.get("label")
    if label_raw is None:
        label_raw = example.raw.get("label")
    gold_letter = _label_to_letter(label_raw, scorer.num_choices)
    predicted = _extract_letter(prediction, scorer.num_choices)
    notes = "letter-match"
    if predicted is None:
        choices = extract_choices(example, scorer.num_choices) or []
        predicted = _extract_by_choice_overlap(prediction, choices, scorer.num_choices)
        notes = "choice-overlap" if predicted is not None else "no-letter-found"
    return ScoreResult(
        correct=(predicted == gold_letter),
        predicted=predicted,
        gold=gold_letter,
        notes=notes,
    )


_GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_TRAILING_NUMBER_RE = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)\D*$")


def _parse_number(text: str) -> str | None:
    m = _GSM8K_FINAL_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    m2 = _TRAILING_NUMBER_RE.search(text.strip())
    if m2:
        return m2.group(1).replace(",", "")
    return None


def _score_gsm8k_final(prediction: str, example: Example) -> ScoreResult:
    gold_raw = example.fields.get("answer")
    if gold_raw is None:
        gold_raw = example.raw.get("answer", "")
    gold_str = str(gold_raw)
    gold_num = _parse_number(gold_str) or ""
    pred_num = _parse_number(prediction) or ""
    correct = bool(gold_num) and pred_num == gold_num
    return ScoreResult(correct=correct, predicted=pred_num or None, gold=gold_num, notes="gsm8k")


def _score_exact_normalize(prediction: str, example: Example) -> ScoreResult:
    gold_raw = example.fields.get("answer")
    if gold_raw is None:
        gold_raw = example.raw.get("answer", "")
    gold_norm = _normalize(str(gold_raw))
    pred_norm = _normalize(prediction)
    return ScoreResult(
        correct=bool(gold_norm) and pred_norm == gold_norm,
        predicted=pred_norm or None,
        gold=gold_norm,
        notes="exact-normalize",
    )


def score(prediction: str, example: Example, task: BenchmarkTask) -> ScoreResult:
    name = task.scorer.name
    if name == "multiple_choice_letter":
        return _score_multiple_choice_letter(prediction, example, task, task.scorer)
    if name == "gsm8k_final":
        return _score_gsm8k_final(prediction, example)
    if name == "exact_normalize":
        return _score_exact_normalize(prediction, example)
    raise ValueError(f"unsupported scorer name: {name!r}")
