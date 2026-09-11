from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    path: Path | None = None
    root: Path | None = None
    splits: dict[str, str] = field(default_factory=dict)
    dataset_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptSpec:
    type: str
    template: str
    field_map: dict[str, str]
    system: str | None = None


@dataclass(frozen=True)
class ScorerSpec:
    name: str
    num_choices: int = 4


@dataclass(frozen=True)
class TaskDefaults:
    max_new_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_examples: int | None = None
    fewshot: int = 0


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    description: str
    source: SourceSpec
    prompt: PromptSpec
    scorer: ScorerSpec
    defaults: TaskDefaults
    spec_path: Path


def _require_dict(obj: object, where: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"expected mapping for {where}, got {type(obj).__name__}")
    return obj


def _require_str(obj: object, where: str) -> str:
    if not isinstance(obj, str) or not obj.strip():
        raise ValueError(f"expected non-empty string for {where}")
    return obj


def _opt_str(obj: object, where: str) -> str | None:
    if obj is None:
        return None
    if not isinstance(obj, str):
        raise ValueError(f"expected string or null for {where}")
    return obj


def _opt_int(obj: object, where: str) -> int | None:
    if obj is None:
        return None
    if isinstance(obj, bool) or not isinstance(obj, int):
        raise ValueError(f"expected integer or null for {where}")
    return int(obj)


def _opt_float(obj: object, where: str) -> float | None:
    if obj is None:
        return None
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        raise ValueError(f"expected number or null for {where}")
    return float(obj)


def _resolve_rel(spec_path: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (spec_path.parent / p).resolve()


def _parse_source(raw: dict[str, Any], spec_path: Path) -> SourceSpec:
    kind = _require_str(raw.get("kind"), "source.kind")
    opts_raw = raw.get("options") or {}
    opts_dict = _require_dict(opts_raw, "source.options")
    dataset_options = {str(k): v for k, v in opts_dict.items()}
    if kind == "jsonl":
        path = _resolve_rel(spec_path, _require_str(raw.get("path"), "source.path"))
        return SourceSpec(kind=kind, path=path, dataset_options=dataset_options)
    if kind == "mmlu_csv":
        root = _resolve_rel(spec_path, _require_str(raw.get("root"), "source.root"))
        splits_raw = raw.get("splits") or {"dev": "dev", "test": "test"}
        splits = {str(k): str(v) for k, v in _require_dict(splits_raw, "source.splits").items()}
        return SourceSpec(kind=kind, root=root, splits=splits, dataset_options=dataset_options)
    if kind == "hellaswag":
        root = _resolve_rel(spec_path, _require_str(raw.get("root"), "source.root"))
        splits_raw = raw.get("splits") or {"val": "hellaswag_val.jsonl"}
        splits = {str(k): str(v) for k, v in _require_dict(splits_raw, "source.splits").items()}
        if not splits:
            raise ValueError("source.splits for hellaswag must be non-empty")
        return SourceSpec(kind=kind, root=root, splits=splits, dataset_options=dataset_options)
    raise ValueError(f"unsupported source.kind: {kind!r}")


def _parse_prompt(raw: dict[str, Any]) -> PromptSpec:
    ptype = _require_str(raw.get("type"), "prompt.type")
    template = _require_str(raw.get("template"), "prompt.template")
    field_map_raw = raw.get("field_map") or {}
    field_map = {
        str(k): str(v) for k, v in _require_dict(field_map_raw, "prompt.field_map").items()
    }
    system_raw = raw.get("system")
    system = None
    if isinstance(system_raw, str) and system_raw.strip() != "":
        system = system_raw
    return PromptSpec(type=ptype, template=template, field_map=field_map, system=system)


def _parse_scorer(raw: dict[str, Any]) -> ScorerSpec:
    name = _require_str(raw.get("name"), "scorer.name")
    nc_raw = raw.get("num_choices", 4)
    nc = _opt_int(nc_raw, "scorer.num_choices") or 4
    if nc < 2:
        raise ValueError("scorer.num_choices must be >= 2")
    return ScorerSpec(name=name, num_choices=nc)


def _parse_defaults(raw: dict[str, Any] | None) -> TaskDefaults:
    if raw is None:
        return TaskDefaults()
    raw = _require_dict(raw, "defaults")
    return TaskDefaults(
        max_new_tokens=_opt_int(raw.get("max_new_tokens"), "defaults.max_new_tokens") or 256,
        temperature=_opt_float(raw.get("temperature"), "defaults.temperature"),
        top_p=_opt_float(raw.get("top_p"), "defaults.top_p"),
        top_k=_opt_int(raw.get("top_k"), "defaults.top_k"),
        max_examples=_opt_int(raw.get("max_examples"), "defaults.max_examples"),
        fewshot=_opt_int(raw.get("fewshot"), "defaults.fewshot") or 0,
    )


def load_task(spec_path: Path) -> BenchmarkTask:
    spec_path = spec_path.expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(f"task spec not found: {spec_path}")
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse YAML at {spec_path}: {exc}") from exc
    raw = _require_dict(raw, f"top level of {spec_path.name}")
    task_id = _require_str(raw.get("id"), "id")
    description = _opt_str(raw.get("description"), "description") or ""
    source = _parse_source(_require_dict(raw.get("source"), "source"), spec_path)
    prompt = _parse_prompt(_require_dict(raw.get("prompt"), "prompt"))
    scorer = _parse_scorer(_require_dict(raw.get("scorer"), "scorer"))
    defaults = _parse_defaults(raw.get("defaults"))
    return BenchmarkTask(
        id=task_id,
        description=description,
        source=source,
        prompt=prompt,
        scorer=scorer,
        defaults=defaults,
        spec_path=spec_path,
    )


def find_task(data_dir: Path, task_id: str) -> Path:
    data_dir = data_dir.expanduser().resolve()
    candidates = [
        data_dir / f"{task_id}.yaml",
        data_dir / f"{task_id}.yml",
        data_dir / task_id / "task.yaml",
        data_dir / task_id / "task.yml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no task spec for id {task_id!r} in {data_dir} "
        f"(looked for {', '.join(c.name for c in candidates)})"
    )


def list_task_ids(data_dir: Path) -> list[str]:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        return []
    ids: set[str] = set()
    for p in data_dir.iterdir():
        if p.is_file() and p.suffix in (".yaml", ".yml"):
            ids.add(p.stem)
        elif p.is_dir():
            for stem in ("task.yaml", "task.yml"):
                if (p / stem).is_file():
                    ids.add(p.name)
                    break
    return sorted(ids)
