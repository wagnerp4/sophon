# Multi-stage SFT

One job is still one SFT pass. A pipeline is several jobs that **continue the same LoRA**. Nothing is merged into the base checkpoint. Rank (`lora_r`) must stay constant after the first adapter exists.

## YAML stages

[`config/finetune/qa-chain.yaml`](../../config/finetune/qa-chain.yaml):

```yaml
preset: llama2_7b_chat
output_name: qa-chain
stages:
  - id: gsm8k
    dataset: gsm8k_instructions
  - id: hellaswag
    dataset: hellaswag
    learning_rate: 1.0e-4
  - id: mmlu
    dataset: mmlu
    learning_rate: 5.0e-5
```

Run with `/finetune from config/finetune/qa-chain.yaml` or `sophon-finetune --recipe config/finetune/qa-chain.yaml`.

Each stage writes its own directory under `data/training/adapters/<preset>/<dataset>/<run_id>/` and an index name `{output_name}-{stage_id}`. `run_meta.json` stores `parent_adapter` and `stage_id`. Failure stops the pipeline. Completed stages stay on disk.

Stage keys overlay the parent recipe. Do not set a different `lora_r` after stage 1.

## Chat continue_from

One extra stage on an existing adapter:

```text
/finetune hellaswag on llama2_7b_chat continue_from=qa-chain-gsm8k
```

`continue_from` accepts index name, `run_id`, or a directory. The job loads `PeftModel.from_pretrained(..., is_trainable=True)` and trains those weights. It does not stack a second LoRA.

There is no DPO path and no CPT (`checkpoints_root` is unused).
