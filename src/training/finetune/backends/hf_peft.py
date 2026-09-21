from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.shared import bitsandbytes_config
from training.common.types import LogCallback, ProgressCallback
from training.finetune.metrics import make_train_callback
from training.finetune.protocols import FinetuneBackend, PreparedModel, TrainResult
from training.finetune.recipe import FinetuneRecipe


def _require_finetune_deps() -> None:
    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "finetune dependencies missing. Run: uv sync --extra finetune"
        ) from exc


def _log(on_log: LogCallback | None, message: str) -> None:
    if on_log is not None:
        on_log(message)
        return
    print(f"sophon: {message}", flush=True)


def _apply_lora(model: object, recipe: FinetuneRecipe, *, on_log: LogCallback | None) -> object:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if recipe.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_kw: dict[str, object] = {
        "r": recipe.lora_r,
        "lora_alpha": recipe.lora_alpha,
        "lora_dropout": recipe.lora_dropout,
        "target_modules": list(recipe.lora_target_modules),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    try:
        model = get_peft_model(model, LoraConfig(**lora_kw))
    except ValueError as exc:
        _log(on_log, f"LoRA target_modules missed ({exc}); falling back to all-linear")
        lora_kw["target_modules"] = "all-linear"
        model = get_peft_model(model, LoraConfig(**lora_kw))
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


class HfPeftFinetuneBackend:
    backend_id = "hf_peft"

    def prepare(
        self,
        base_model_path: Path,
        recipe: FinetuneRecipe,
        *,
        on_log: LogCallback | None = None,
        adapter_path: Path | None = None,
    ) -> PreparedModel:
        _require_finetune_deps()
        from peft import PeftModel

        if not torch.cuda.is_available():
            raise RuntimeError("hf_peft finetune requires CUDA (torch.cuda.is_available() must be True)")

        _log(on_log, f"loading base model from {base_model_path} (4bit={recipe.load_in_4bit})")
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_mode = "4bit" if recipe.load_in_4bit else "none"
        bnb = bitsandbytes_config(quant_mode)
        model_kw: dict[str, object] = {
            "local_files_only": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if bnb is not None:
            model_kw["quantization_config"] = bnb

        model = AutoModelForCausalLM.from_pretrained(str(base_model_path), **model_kw)
        if adapter_path is not None:
            _log(on_log, f"continuing LoRA from {adapter_path}")
            try:
                model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
            except TypeError:
                model = PeftModel.from_pretrained(model, str(adapter_path))
                if hasattr(model, "train"):
                    model.train()
        else:
            model = _apply_lora(model, recipe, on_log=on_log)
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
        _require_finetune_deps()
        from trl import SFTConfig, SFTTrainer

        model = prepared.model
        tokenizer = prepared.tokenizer
        model.train()

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
            "fp16": not use_bf16,
            "bf16": use_bf16,
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
            sft_config = SFTConfig(**sft_kw)
        except TypeError:
            if "eval_strategy" in sft_kw:
                sft_kw["evaluation_strategy"] = sft_kw.pop("eval_strategy")
                sft_config = SFTConfig(**sft_kw)
            else:
                raise

        _log(on_log, "starting SFTTrainer (hf_peft)")
        trainer_kw: dict[str, object] = {
            "model": model,
            "processing_class": tokenizer,
            "train_dataset": train_dataset,
            "args": sft_config,
        }
        if eval_dataset is not None:
            trainer_kw["eval_dataset"] = eval_dataset
        trainer = SFTTrainer(**trainer_kw)
        trainer.add_callback(
            make_train_callback(metrics_path, on_progress=on_progress, on_log=on_log)
        )
        train_result = trainer.train()
        metrics = dict(train_result.metrics or {})
        model.eval()
        return TrainResult(metrics=metrics, adapter_dir=output_dir)

    def save_adapter(self, prepared: PreparedModel, adapter_dir: Path) -> Path:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        prepared.model.save_pretrained(str(adapter_dir))
        prepared.tokenizer.save_pretrained(str(adapter_dir))
        return adapter_dir
