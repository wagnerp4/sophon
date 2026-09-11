from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.shared import bitsandbytes_config
from training.common.types import LogCallback, ProgressCallback
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
    print(f"orodruin: {message}", flush=True)


class HfPeftFinetuneBackend:
    backend_id = "hf_peft"

    def prepare(
        self,
        base_model_path: Path,
        recipe: FinetuneRecipe,
        *,
        on_log: LogCallback | None = None,
    ) -> PreparedModel:
        _require_finetune_deps()
        from peft import LoraConfig, get_peft_model

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
        lora_config = LoraConfig(
            r=recipe.lora_r,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=recipe.lora_dropout,
            target_modules=list(recipe.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return PreparedModel(model=model, tokenizer=tokenizer, backend_id=self.backend_id)

    def train(
        self,
        prepared: PreparedModel,
        train_dataset: object,
        recipe: FinetuneRecipe,
        output_dir: Path,
        *,
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

        sft_config = SFTConfig(
            output_dir=str(train_output),
            per_device_train_batch_size=recipe.per_device_train_batch_size,
            gradient_accumulation_steps=recipe.gradient_accumulation_steps,
            warmup_steps=recipe.warmup_steps,
            num_train_epochs=recipe.num_train_epochs,
            learning_rate=recipe.learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=recipe.logging_steps,
            optim=recipe.optim,
            weight_decay=recipe.weight_decay,
            lr_scheduler_type=recipe.lr_scheduler_type,
            seed=recipe.seed,
            max_length=recipe.max_seq_length,
            dataset_text_field="text",
            gradient_checkpointing=True,
            report_to=[],
        )

        _log(on_log, "starting SFTTrainer (hf_peft)")
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=sft_config,
        )

        class _ProgressCallback:
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None or on_progress is None:
                    return
                step = int(getattr(state, "global_step", 0) or 0)
                max_steps = int(getattr(state, "max_steps", 0) or 0)
                loss = logs.get("loss")
                label = f"step {step}"
                if loss is not None:
                    label = f"step {step} loss={loss:.4f}"
                on_progress(step, max_steps, label)

        trainer.add_callback(_ProgressCallback())
        train_result = trainer.train()
        metrics = dict(train_result.metrics or {})
        model.eval()
        return TrainResult(metrics=metrics, adapter_dir=output_dir)

    def save_adapter(self, prepared: PreparedModel, adapter_dir: Path) -> Path:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        prepared.model.save_pretrained(str(adapter_dir))
        prepared.tokenizer.save_pretrained(str(adapter_dir))
        return adapter_dir
