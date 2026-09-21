from __future__ import annotations

from pathlib import Path

import torch

from training.common.types import LogCallback, ProgressCallback
from training.finetune.metrics import make_train_callback
from training.finetune.protocols import PreparedModel, TrainResult
from training.finetune.recipe import FinetuneRecipe


def _require_unsloth() -> None:
    try:
        import unsloth  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth backend not available. Run: uv sync --extra finetune-unsloth"
        ) from exc


def _log(on_log: LogCallback | None, message: str) -> None:
    if on_log is not None:
        on_log(message)
        return
    print(f"sophon: {message}", flush=True)


class UnslothFinetuneBackend:
    backend_id = "unsloth"

    def prepare(
        self,
        base_model_path: Path,
        recipe: FinetuneRecipe,
        *,
        on_log: LogCallback | None = None,
        adapter_path: Path | None = None,
    ) -> PreparedModel:
        _require_unsloth()
        from peft import PeftModel
        from unsloth import FastLanguageModel

        if not torch.cuda.is_available():
            raise RuntimeError("unsloth finetune requires CUDA")

        _log(on_log, f"loading base model via Unsloth from {base_model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(base_model_path),
            max_seq_length=recipe.max_seq_length,
            dtype=None,
            load_in_4bit=recipe.load_in_4bit,
        )
        if adapter_path is not None:
            _log(on_log, f"continuing LoRA from {adapter_path}")
            try:
                model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
            except TypeError:
                model = PeftModel.from_pretrained(model, str(adapter_path))
                if hasattr(model, "train"):
                    model.train()
        else:
            try:
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=recipe.lora_r,
                    target_modules=list(recipe.lora_target_modules),
                    lora_alpha=recipe.lora_alpha,
                    lora_dropout=recipe.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=recipe.seed,
                )
            except ValueError as exc:
                _log(on_log, f"LoRA target_modules missed ({exc}); falling back to all-linear")
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=recipe.lora_r,
                    target_modules="all-linear",
                    lora_alpha=recipe.lora_alpha,
                    lora_dropout=recipe.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=recipe.seed,
                )
        return PreparedModel(model=model, tokenizer=tokenizer, backend_id=self.backend_id)

    def train(
        self,
        prepared: PreparedModel,
        train_dataset: object,
        recipe: FinetuneRecipe,
        output_dir: Path,
        *,
        eval_dataset: object | None = None,
        on_progress: ProgressCallback | None = None,
        on_log: LogCallback | None = None,
    ) -> TrainResult:
        _require_unsloth()
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel, is_bfloat16_supported

        model = prepared.model
        tokenizer = prepared.tokenizer

        try:
            FastLanguageModel.for_training(model)
        except AttributeError:
            model.train()

        train_output = output_dir / "trainer_output"
        train_output.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.jsonl"

        sft_kw: dict[str, object] = {
            "output_dir": str(train_output),
            "per_device_train_batch_size": recipe.per_device_train_batch_size,
            "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
            "warmup_steps": recipe.warmup_steps,
            "num_train_epochs": recipe.num_train_epochs,
            "learning_rate": recipe.learning_rate,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            "logging_steps": recipe.logging_steps,
            "optim": recipe.optim,
            "weight_decay": recipe.weight_decay,
            "lr_scheduler_type": recipe.lr_scheduler_type,
            "seed": recipe.seed,
            "max_length": recipe.max_seq_length,
            "dataset_text_field": "text",
            "gradient_checkpointing": True,
            "report_to": [],
        }
        if eval_dataset is not None:
            sft_kw["eval_strategy"] = "steps"
            sft_kw["eval_steps"] = recipe.eval_steps or recipe.logging_steps
            sft_kw["per_device_eval_batch_size"] = recipe.per_device_train_batch_size

        try:
            sft_args = SFTConfig(**sft_kw)
        except TypeError:
            if "eval_strategy" in sft_kw:
                sft_kw["evaluation_strategy"] = sft_kw.pop("eval_strategy")
                sft_args = SFTConfig(**sft_kw)
            else:
                raise

        _log(on_log, "starting SFTTrainer (unsloth)")
        trainer_kw: dict[str, object] = {
            "model": model,
            "processing_class": tokenizer,
            "train_dataset": train_dataset,
            "args": sft_args,
        }
        if eval_dataset is not None:
            trainer_kw["eval_dataset"] = eval_dataset
        trainer = SFTTrainer(**trainer_kw)
        trainer.add_callback(
            make_train_callback(metrics_path, on_progress=on_progress, on_log=on_log)
        )
        train_result = trainer.train()

        try:
            FastLanguageModel.for_inference(model)
        except AttributeError:
            model.eval()

        metrics = dict(train_result.metrics or {})
        return TrainResult(metrics=metrics, adapter_dir=output_dir)

    def save_adapter(self, prepared: PreparedModel, adapter_dir: Path) -> Path:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        prepared.model.save_pretrained(str(adapter_dir))
        prepared.tokenizer.save_pretrained(str(adapter_dir))
        return adapter_dir
