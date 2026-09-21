from __future__ import annotations

import json
import time
from pathlib import Path

from training.common.types import LogCallback, ProgressCallback


def append_metrics_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def make_train_callback(
    metrics_path: Path,
    *,
    on_progress: ProgressCallback | None = None,
    on_log: LogCallback | None = None,
    started_at: float | None = None,
):
    from transformers import TrainerCallback

    t0 = started_at if started_at is not None else time.time()

    class _TrainMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            step = int(getattr(state, "global_step", 0) or 0)
            max_steps = int(getattr(state, "max_steps", 0) or 0)
            loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            lr = logs.get("learning_rate")
            epoch = logs.get("epoch")
            row: dict[str, object] = {
                "step": step,
                "loss": float(loss) if isinstance(loss, (int, float)) else None,
                "eval_loss": float(eval_loss) if isinstance(eval_loss, (int, float)) else None,
                "learning_rate": float(lr) if isinstance(lr, (int, float)) else None,
                "epoch": float(epoch) if isinstance(epoch, (int, float)) else None,
                "wall_ms": int((time.time() - t0) * 1000),
            }
            append_metrics_row(metrics_path, row)
            label = f"step {step}"
            if isinstance(loss, (int, float)):
                label = f"step {step} loss={float(loss):.4f}"
            if isinstance(eval_loss, (int, float)):
                label = f"{label} eval_loss={float(eval_loss):.4f}"
            if on_progress is not None:
                on_progress(step, max_steps, label)
            elif on_log is not None:
                on_log(label)

    return _TrainMetricsCallback()
