from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.hf.backend import (
    GenerationResult,
    ModelMeta,
    generate_response,
    load_processor_and_model,
    parsed_to_display_text,
    read_model_meta,
)
from backend.hf.paths import require_model_on_disk, resolve_local_model_dir
from backend.ollama.backend import (
    chat_complete as ollama_chat_complete,
    ollama_base_url,
    ping_daemon as ollama_ping,
    verify_model_known as ollama_verify_model,
)
from eval.load_examples import Example, iter_examples, render_prompt
from eval.scorers import score
from eval.task_spec import BenchmarkTask


@dataclass
class RunConfig:
    backend: str = "hf"
    model_path: str | None = None
    quantization: str = "none"
    enable_thinking: bool = False
    ollama_model: str | None = None
    ollama_base_url: str | None = None
    server_model: str | None = None
    lmstudio_base_url: str | None = None
    max_new_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    split: str | None = None
    limit: int | None = None
    out_dir: Path = field(default_factory=lambda: Path("evaluation_runs"))
    run_id: str | None = None
    run_label: str | None = None


@dataclass
class TaskResult:
    run_id: str
    task_id: str
    total: int
    correct: int
    accuracy: float
    elapsed_s: float
    config: dict[str, Any]
    summary_path: Path
    predictions_path: Path


def _server_model_name(config: RunConfig) -> str:
    return str(config.server_model or config.ollama_model or "").strip()


def _now_label() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _serialize_config(task: BenchmarkTask, config: RunConfig) -> dict[str, Any]:
    cfg = asdict(config)
    cfg["out_dir"] = str(config.out_dir)
    cfg["task_id"] = task.id
    cfg["task_spec_path"] = str(task.spec_path)
    cfg["task_description"] = task.description
    cfg["scorer"] = task.scorer.name
    cfg["scorer_num_choices"] = task.scorer.num_choices
    return cfg


def _resolved_generation_params(task: BenchmarkTask, config: RunConfig, meta: ModelMeta | None) -> dict[str, Any]:
    max_new = config.max_new_tokens if config.max_new_tokens is not None else task.defaults.max_new_tokens
    temperature = config.temperature
    if temperature is None:
        temperature = task.defaults.temperature
    if temperature is None and meta is not None:
        temperature = meta.default_temperature
    top_p = config.top_p if config.top_p is not None else task.defaults.top_p
    if top_p is None and meta is not None:
        top_p = meta.default_top_p
    top_k = config.top_k if config.top_k is not None else task.defaults.top_k
    if top_k is None and meta is not None:
        top_k = meta.default_top_k
    return {
        "max_new_tokens": int(max_new),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


def _build_messages(task: BenchmarkTask, example: Example) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    if task.prompt.system:
        messages.append({"role": "system", "content": task.prompt.system})
    messages.append({"role": "user", "content": render_prompt(task, example)})
    return messages


def _open_jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _emit(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _hf_generate(
    processor: object,
    model: object,
    meta: ModelMeta,
    messages: list[dict[str, object]],
    *,
    gen_params: dict[str, Any],
    enable_thinking: bool,
    repetition_penalty: float | None,
    seed: int | None,
) -> tuple[str, GenerationResult]:
    result = generate_response(
        processor,
        model,
        messages,
        max_new_tokens=gen_params["max_new_tokens"],
        enable_thinking=enable_thinking,
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        top_k=gen_params["top_k"],
        repetition_penalty=repetition_penalty,
        seed=seed,
        strip=True,
        extra_specials=meta.special_tokens,
        eos_token_ids=meta.eos_token_ids or None,
    )
    return parsed_to_display_text(result.parsed), result


def _ollama_generate(
    model_name: str,
    messages: list[dict[str, object]],
    *,
    gen_params: dict[str, Any],
    base_url: str | None,
) -> tuple[str, float]:
    t0 = time.perf_counter()
    text = ollama_chat_complete(
        model_name,
        messages,
        max_new_tokens=gen_params["max_new_tokens"],
        base_url=base_url,
    )
    return text, max(0.0, time.perf_counter() - t0)


def _lmstudio_generate(
    model_name: str,
    messages: list[dict[str, object]],
    *,
    gen_params: dict[str, Any],
    base_url: str | None,
) -> tuple[str, float]:
    from backend.lmstudio.backend import chat_complete as lmstudio_chat_complete

    t0 = time.perf_counter()
    result = lmstudio_chat_complete(
        model_name,
        messages,
        max_new_tokens=gen_params["max_new_tokens"],
        temperature=gen_params.get("temperature"),
        top_p=gen_params.get("top_p"),
        base_url=base_url,
    )
    text = str(getattr(result, "text", None) or result or "")
    return text, max(0.0, time.perf_counter() - t0)


def _server_generate(
    backend: str,
    model_name: str,
    messages: list[dict[str, object]],
    *,
    gen_params: dict[str, Any],
    ollama_base_url: str | None,
    lmstudio_base_url: str | None,
) -> tuple[str, float]:
    if backend == "ollama":
        return _ollama_generate(
            model_name,
            messages,
            gen_params=gen_params,
            base_url=ollama_base_url,
        )
    if backend == "lmstudio":
        return _lmstudio_generate(
            model_name,
            messages,
            gen_params=gen_params,
            base_url=lmstudio_base_url,
        )
    raise ValueError(f"unsupported server backend: {backend!r}")


def _prepare_hf(config: RunConfig) -> tuple[object, object, ModelMeta]:
    model_dir = require_model_on_disk(resolve_local_model_dir(config.model_path))
    processor, model = load_processor_and_model(model_dir, config.quantization)
    meta = read_model_meta(model_dir, processor)
    return processor, model, meta


def _prepare_ollama(config: RunConfig) -> None:
    name = _server_model_name(config)
    if not name:
        raise ValueError("--ollama-model / server_model is required when --backend ollama")
    base = config.ollama_base_url or ollama_base_url()
    ollama_ping(base_url=base)
    ollama_verify_model(name, base_url=base)


def _prepare_lmstudio(config: RunConfig) -> None:
    from backend.lmstudio.backend import list_model_names, lmstudio_base_url, ping_daemon

    name = _server_model_name(config)
    if not name:
        raise ValueError("server_model is required when backend=lmstudio")
    base = config.lmstudio_base_url or lmstudio_base_url()
    if not ping_daemon(base_url=base):
        raise RuntimeError(f"LM Studio not reachable at {base}")
    known = list_model_names(base_url=base)
    if known and name not in known:
        raise ValueError(f"LM Studio model {name!r} not in loaded models: {known}")


def run_task(task: BenchmarkTask, config: RunConfig) -> TaskResult:
    rid = config.run_id or _now_label()
    task_out_dir = (config.out_dir / rid / task.id).resolve()
    task_out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = task_out_dir / "predictions.jsonl"
    summary_path = task_out_dir / "summary.json"

    effective_limit = config.limit if config.limit is not None else task.defaults.max_examples

    cfg_record = _serialize_config(task, config)
    cfg_record["effective_limit"] = effective_limit
    cfg_record["run_id"] = rid

    meta: ModelMeta | None = None
    processor: object | None = None
    model: object | None = None
    if config.backend == "hf":
        processor, model, meta = _prepare_hf(config)
    elif config.backend == "ollama":
        _prepare_ollama(config)
    elif config.backend == "lmstudio":
        _prepare_lmstudio(config)
    else:
        raise ValueError(f"unsupported backend: {config.backend!r}")

    gen_params = _resolved_generation_params(task, config, meta)
    cfg_record["generation"] = {
        "max_new_tokens": gen_params["max_new_tokens"],
        "temperature": gen_params["temperature"],
        "top_p": gen_params["top_p"],
        "top_k": gen_params["top_k"],
        "repetition_penalty": config.repetition_penalty,
        "seed": config.seed,
    }
    _emit(
        f"sophon-benchmark: task={task.id} run_id={rid} backend={config.backend} limit={effective_limit} "
        f"out={task_out_dir}"
    )

    correct = 0
    total = 0
    total_gen_s = 0.0
    t_start = time.perf_counter()

    with _open_jsonl_writer(predictions_path) as fh:
        for example in iter_examples(task, split=config.split, limit=effective_limit):
            messages = _build_messages(task, example)
            try:
                if config.backend == "hf":
                    assert processor is not None and model is not None and meta is not None
                    text, gen_result = _hf_generate(
                        processor,
                        model,
                        meta,
                        messages,
                        gen_params=gen_params,
                        enable_thinking=config.enable_thinking,
                        repetition_penalty=config.repetition_penalty,
                        seed=config.seed,
                    )
                    gen_s = gen_result.gen_time_s
                    extra = {
                        "input_tokens": gen_result.input_tokens,
                        "new_tokens": gen_result.new_tokens,
                        "stop_reason": gen_result.stop_reason,
                    }
                else:
                    text, gen_s = _server_generate(
                        config.backend,
                        _server_model_name(config),
                        messages,
                        gen_params=gen_params,
                        ollama_base_url=config.ollama_base_url,
                        lmstudio_base_url=config.lmstudio_base_url,
                    )
                    extra = {}
            except RuntimeError as exc:
                _emit(f"  example {example.idx}: backend error: {exc}")
                continue

            res = score(text, example, task)
            total += 1
            if res.correct:
                correct += 1
            total_gen_s += gen_s

            record: dict[str, Any] = {
                "idx": example.idx,
                "subject": example.subject,
                "predicted_letter": res.predicted,
                "gold": res.gold,
                "correct": res.correct,
                "notes": res.notes,
                "response": text,
                "gen_time_s": gen_s,
            }
            if extra:
                record.update(extra)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            if total % 10 == 0:
                acc = correct / total
                _emit(f"  {total} done; running accuracy {acc:.3f}")

    elapsed_s = max(0.0, time.perf_counter() - t_start)
    accuracy = (correct / total) if total > 0 else 0.0
    summary = {
        **cfg_record,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_s": elapsed_s,
        "total_generation_s": total_gen_s,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(
        f"sophon-benchmark: task={task.id} done in {elapsed_s:.1f}s "
        f"accuracy={accuracy:.3f} ({correct}/{total}) -> {summary_path}"
    )

    return TaskResult(
        run_id=rid,
        task_id=task.id,
        total=total,
        correct=correct,
        accuracy=accuracy,
        elapsed_s=elapsed_s,
        config=cfg_record,
        summary_path=summary_path,
        predictions_path=predictions_path,
    )
