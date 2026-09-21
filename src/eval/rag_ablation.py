from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from eval.load_examples import Example, iter_examples, render_prompt
from eval.runner import (
    RunConfig,
    TaskResult,
    _hf_generate,
    _prepare_hf,
    _prepare_lmstudio,
    _prepare_ollama,
    _resolved_generation_params,
    _serialize_config,
    _server_generate,
    _server_model_name,
)
from eval.scorers import ScoreResult, _normalize
from eval.task_spec import BenchmarkTask
from processing.text.context.builder import build_messages_for_model
from processing.text.retrieval import RetrievalQuery, load_rag_retriever
from processing.text.retrieval.adaptive import decide_retrieval
from processing.text.retrieval.corpus import default_index_exists, default_leann_index_path
from processing.text.retrieval.types import RetrievalResult


EmitFn = Callable[[str], None]


def _score_contains(prediction: str, gold: str) -> ScoreResult:
    gold_norm = _normalize(gold)
    pred_norm = _normalize(prediction)
    if not gold_norm:
        return ScoreResult(correct=False, predicted=pred_norm or None, gold="", notes="missing-gold")
    ok = gold_norm in pred_norm
    return ScoreResult(correct=ok, predicted=pred_norm or None, gold=gold_norm, notes="contains-normalize")


def _gold_from_example(example: Example) -> str:
    for key in ("gold", "answer"):
        if key in example.fields and example.fields[key] is not None:
            return str(example.fields[key])
        if key in example.raw and example.raw[key] is not None:
            return str(example.raw[key])
    return ""


def _retrieve_context(query: str, *, index_path: str, top_k: int) -> tuple[list[dict[str, object]], dict[str, Any]]:
    retriever = load_rag_retriever("leann", native_index_path=index_path)
    if retriever.backend_id() == "noop":
        return [], {"error": "noop_retriever", "n_hits": 0}
    adaptive = decide_retrieval(query, structure_available=False)
    meta: dict[str, Any] = {
        "adaptive": {"decision": adaptive.decision.value, "reason": adaptive.reason},
    }
    if adaptive.decision.value == "skip":
        meta["n_hits"] = 0
        return [], meta
    result = retriever.retrieve(RetrievalQuery(text=query, params={"top_k": top_k}))
    metrics = result.extras.get("metrics") if isinstance(result.extras, dict) else {}
    meta["metrics"] = metrics if isinstance(metrics, dict) else {}
    meta["n_hits"] = len(result.chunks)
    return list(result.chunks), meta


def run_rag_ablation(
    task: BenchmarkTask,
    config: RunConfig,
    *,
    index_path: str | None = None,
    top_k: int = 5,
    on_emit: EmitFn | None = None,
) -> TaskResult:
    """Run each example twice: no retrieval vs LEANN context. Compare contain-match + latency."""

    def emit(msg: str) -> None:
        if on_emit is not None:
            on_emit(msg)
        else:
            print(msg, flush=True)

    rid = config.run_id or time.strftime("%Y%m%d_%H%M%S")
    task_out_dir = (config.out_dir / rid / task.id).resolve()
    task_out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = task_out_dir / "predictions.jsonl"
    summary_path = task_out_dir / "summary.json"
    index = index_path or str(default_leann_index_path())
    if not default_index_exists(Path(index)):
        raise ValueError(
            f"LEANN index missing at {index}. Run /rag-index --rebuild "
            "(set SOPHON_VAULT_PATH) before /eval rag."
        )
    effective_limit = config.limit if config.limit is not None else task.defaults.max_examples

    cfg_record = _serialize_config(task, config)
    cfg_record["effective_limit"] = effective_limit
    cfg_record["run_id"] = rid
    cfg_record["rag_index"] = index
    cfg_record["rag_top_k"] = top_k

    meta = None
    processor = None
    model = None
    if config.backend == "hf":
        processor, model, meta = _prepare_hf(config)
    elif config.backend == "ollama":
        _prepare_ollama(config)
    elif config.backend == "lmstudio":
        _prepare_lmstudio(config)
    else:
        raise ValueError(f"unsupported backend: {config.backend!r}")

    gen_params = _resolved_generation_params(task, config, meta)
    cfg_record["generation"] = gen_params

    emit(f"rag_ablation: index={index} limit={effective_limit} backend={config.backend}")

    total = 0
    correct_off = 0
    correct_on = 0
    latency_off = 0.0
    latency_on = 0.0
    hits_on = 0
    t_start = time.perf_counter()
    server_name = _server_model_name(config)

    with predictions_path.open("w", encoding="utf-8") as fh:
        for example in iter_examples(task, split=config.split, limit=effective_limit):
            user_text = render_prompt(task, example)
            gold = _gold_from_example(example)
            base_messages: list[dict[str, object]] = []
            if task.prompt.system:
                base_messages.append({"role": "system", "content": task.prompt.system})
            base_messages.append({"role": "user", "content": user_text})

            t0 = time.perf_counter()
            if config.backend == "hf":
                assert processor is not None and model is not None and meta is not None
                text_off, gen_off = _hf_generate(
                    processor,
                    model,
                    meta,
                    base_messages,
                    gen_params=gen_params,
                    enable_thinking=config.enable_thinking,
                    repetition_penalty=config.repetition_penalty,
                    seed=config.seed,
                )
                off_s = float(gen_off.gen_time_s)
            else:
                text_off, off_s = _server_generate(
                    config.backend,
                    server_name,
                    base_messages,
                    gen_params=gen_params,
                    ollama_base_url=config.ollama_base_url,
                    lmstudio_base_url=config.lmstudio_base_url,
                )
            off_wall = time.perf_counter() - t0
            score_off = (
                _score_contains(text_off, gold)
                if gold
                else ScoreResult(correct=False, predicted=text_off[:200], gold="", notes="no-gold")
            )

            chunks, ret_meta = _retrieve_context(user_text, index_path=index, top_k=top_k)
            retrieval = RetrievalResult(chunks=chunks, extras=dict(ret_meta))
            built = build_messages_for_model(
                [m for m in base_messages if m.get("role") != "system"],
                system_text=task.prompt.system,
                retrieval=retrieval if chunks else None,
                memory_turns=[],
            )
            on_messages = built.messages

            t1 = time.perf_counter()
            if config.backend == "hf":
                assert processor is not None and model is not None and meta is not None
                text_on, gen_on = _hf_generate(
                    processor,
                    model,
                    meta,
                    on_messages,
                    gen_params=gen_params,
                    enable_thinking=config.enable_thinking,
                    repetition_penalty=config.repetition_penalty,
                    seed=config.seed,
                )
                on_s = float(gen_on.gen_time_s)
            else:
                text_on, on_s = _server_generate(
                    config.backend,
                    server_name,
                    on_messages,
                    gen_params=gen_params,
                    ollama_base_url=config.ollama_base_url,
                    lmstudio_base_url=config.lmstudio_base_url,
                )
            on_wall = time.perf_counter() - t1
            score_on = (
                _score_contains(text_on, gold)
                if gold
                else ScoreResult(correct=False, predicted=text_on[:200], gold="", notes="no-gold")
            )

            total += 1
            if score_off.correct:
                correct_off += 1
            if score_on.correct:
                correct_on += 1
            latency_off += off_wall
            latency_on += on_wall
            n_hits = int(ret_meta.get("n_hits") or 0)
            if n_hits > 0:
                hits_on += 1

            row = {
                "idx": example.idx,
                "gold": gold,
                "n_hits": n_hits,
                "retrieval": ret_meta,
                "no_rag": {
                    "text": text_off,
                    "correct": score_off.correct,
                    "latency_s": off_wall,
                    "gen_s": off_s,
                },
                "with_rag": {
                    "text": text_on,
                    "correct": score_on.correct,
                    "latency_s": on_wall,
                    "gen_s": on_s,
                },
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            emit(
                f"  [{example.idx}] hits={n_hits} "
                f"off={'Y' if score_off.correct else 'n'} "
                f"on={'Y' if score_on.correct else 'n'} "
                f"dt_off={off_wall:.2f}s dt_on={on_wall:.2f}s"
            )

    elapsed = time.perf_counter() - t_start
    acc_off = (correct_off / total) if total else 0.0
    acc_on = (correct_on / total) if total else 0.0
    summary = {
        "run_id": rid,
        "task_id": task.id,
        "total": total,
        "correct_no_rag": correct_off,
        "correct_with_rag": correct_on,
        "accuracy_no_rag": acc_off,
        "accuracy_with_rag": acc_on,
        "accuracy": acc_on,
        "correct": correct_on,
        "mean_latency_no_rag_s": (latency_off / total) if total else 0.0,
        "mean_latency_with_rag_s": (latency_on / total) if total else 0.0,
        "retrieval_hit_rate": (hits_on / total) if total else 0.0,
        "elapsed_s": elapsed,
        "config": cfg_record,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(
        f"rag_ablation done: n={total} acc_off={acc_off:.3f} acc_on={acc_on:.3f} "
        f"hit_rate={summary['retrieval_hit_rate']:.3f}"
    )
    return TaskResult(
        run_id=rid,
        task_id=task.id,
        total=total,
        correct=correct_on,
        accuracy=acc_on,
        elapsed_s=elapsed,
        config=cfg_record,
        summary_path=summary_path,
        predictions_path=predictions_path,
    )
