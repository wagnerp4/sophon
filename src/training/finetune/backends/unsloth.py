from __future__ import annotations

from pathlib import Path

import torch

from training.common.types import LogCallback, ProgressCallback
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
    print(f"orodruin: {message}", flush=True)


class UnslothFinetuneBackend:
    backend_id = "unsloth"

    def prepare(
        self,
        base_model_path: Path,
        recipe: FinetuneRecipe,
        *,
        on_log: LogCallback | None = None,
    ) -> PreparedModel:
        _require_unsloth()
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

        sft_config = SFTConfig(
            output_dir=str(train_output),
            per_device_train_batch_size=recipe.per_device_train_batch_size,
            gradient_accumulation_steps=recipe.gradient_accumulation_steps,
            warmup_steps=recipe.warmup_steps,
            num_train_epochs=recipe.num_train_epochs,
            learning_rate=recipe.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
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

        _log(on_log, "starting SFTTrainer (unsloth)")
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
