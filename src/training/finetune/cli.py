from __future__ import annotations

import argparse
import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from backend.hf.registry import preset_keys_sorted
from training.finetune.backends.factory import finetune_backend_ids
from training.finetune.datasets.registry import dataset_preset_keys_sorted
from training.finetune.job import run_finetune_job
from training.finetune.recipe import parse_recipe_overrides
from utils.device.env_bootstrap import load_orodruin_dotenv, orodruin_project_root


def main(argv: list[str] | None = None) -> None:
    load_orodruin_dotenv()
    parser = argparse.ArgumentParser(description="LoRA finetune a orodruin HF preset on a registered dataset.")
    parser.add_argument(
        "--preset",
        required=True,
        choices=preset_keys_sorted(),
        help="Base model preset key from backend.hf.registry.",
    )
    parser.add_argument(
        "--dataset",
        default="gsm8k_instructions",
        choices=dataset_preset_keys_sorted(),
        help="Finetune dataset preset (default: gsm8k_instructions).",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=list(finetune_backend_ids()),
        help="Training backend (default: auto).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional recipe overrides as KEY=VAL (e.g. max_examples=500 epochs=1 lr=2e-4).",
    )
    args = parser.parse_args(argv)

    def on_log(msg: str) -> None:
        print(f"[orodruin-finetune] {msg}", flush=True)

    def on_progress(step: int, total: int, label: str) -> None:
        if total > 0:
            print(f"[orodruin-finetune] {label} ({step}/{total})", flush=True)
        else:
            print(f"[orodruin-finetune] {label}", flush=True)

    try:
        result = run_finetune_job(
            args.preset,
            args.dataset,
            project_root=orodruin_project_root(),
            backend_id=args.backend,
            recipe_override_tokens=list(args.overrides),
            on_progress=on_progress,
            on_log=on_log,
        )
    except Exception as exc:
        print(f"[orodruin-finetune] failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"[orodruin-finetune] adapter saved to {result.adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
