# Hyperparameters

Defaults live in [`config/finetune/default.yaml`](../../config/finetune/default.yaml) and [`FinetuneRecipe`](../../src/training/finetune/recipe.py). Chat aliases are accepted as `KEY=VAL`. Unknown keys raise. Silent skip is gone.

Load order: `config/finetune/default.yaml` → `config/finetune/datasets/<id>.yaml` if present → `config/finetune/<preset>.yaml` if present → `/finetune from PATH` → `KEY=VAL`.

`/finetune-config` dumps the resolved mapping and the YAML path so you can open it in the editor.

## Fields

- `preset` — HF preset key. Chat: `on PRESET`.
- `dataset` — train loader id. Chat: first positional.
- `output_name` — index name. Stages become `{output_name}-{stage_id}`.
- `max_seq_length` (2048) — TRL `SFTConfig.max_length`.
- `load_in_4bit` (true) — bitsandbytes NF4. Set false for full weights on CUDA.
- `lora_r` / `lora_alpha` (8 / 8) — PEFT rank and scale. Cannot change when `continue_from` is set.
- `lora_dropout` (0.0)
- `lora_target_modules` — default q/k/v/o/gate/up/down_proj. Misses fall back to `all-linear`.
- `per_device_train_batch_size` (2) — alias `batch_size`.
- `gradient_accumulation_steps` (4)
- `warmup_steps` (5)
- `num_train_epochs` (1) — alias `epochs`.
- `learning_rate` (2e-4) — alias `lr`.
- `logging_steps` (10) — also the JSONL / chat step interval.
- `weight_decay` (0.01)
- `lr_scheduler_type` (`linear`)
- `seed` (1)
- `max_examples` (500). `0` means all rows.
- `optim` (`adamw_8bit`)
- `eval_split` — omit for train-only. A fraction such as `0.05` holdouts from the SFT map. A name such as `val` loads that split.
- `eval_steps` — defaults to `logging_steps` when eval is on.
- `report_curve` (true) — TUI sparkline while `/finetune` is running.
- `continue_from` — adapter name, run_id, or directory. Continues the same LoRA.
- `allow_large` (false) — required for `llama3_70b_instruct`.
- `oom_backoff` (false) — one retry at `batch_size=1` with accum scaled up.
- `stages` — list of mappings. See [stages.md](stages.md).

Example:

```text
/finetune gsm8k_instructions on llama2_7b_chat max_examples=200 epochs=1 lr=2e-4 batch_size=1
```
