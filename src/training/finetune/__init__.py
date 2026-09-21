from __future__ import annotations

__all__ = ["FinetuneJobResult", "run_finetune_job"]


def __getattr__(name: str):
    if name in {"FinetuneJobResult", "run_finetune_job"}:
        from training.finetune.job import FinetuneJobResult, run_finetune_job

        values = {
            "FinetuneJobResult": FinetuneJobResult,
            "run_finetune_job": run_finetune_job,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
