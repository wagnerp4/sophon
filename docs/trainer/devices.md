# Devices, RAM, OOM

## Train vs chat

Finetune backends (`hf_peft`, `unsloth`) require `torch.cuda.is_available()`. MPS and CPU are refused. Chat inference may still use MPS or CPU with `quantization=none`. Bitsandbytes 4-bit LoRA does not run on Metal. `device_map=auto` does not change that.

Train on the Windows GPU `.venv` (`C:\Software\Python\NLP\Personal\sophon`). Do not create a WSL CUDA venv for this job. Install extras there: `uv sync --extra finetune`. Optional Unsloth: `--extra finetune-unsloth`. Backend `auto` picks Unsloth when it imports.

## What the job sets

- `device_map=auto`, `low_cpu_mem_usage=true`
- `load_in_4bit=true` by default (NF4, double quant, bfloat16 compute)
- `gradient_checkpointing=true`
- `optim=adamw_8bit`
- `prepare_model_for_kbit_training` before PEFT when 4-bit is on
- After the run: `gc.collect()` and `torch.cuda.empty_cache()`
- Chat unloads the session model before `/finetune`

`device_map=auto` may offload layers to host RAM when VRAM is tight. That can look like a hang rather than an OOM.

## Preflight

The job calls the same snapshot as `sophon-system-check` (CUDA free bytes, host RAM, bitsandbytes import, optional model index `total_size`). It logs batch, accum, `max_seq_length`, and 4-bit. Missing CUDA fails immediately. `llama3_70b_instruct` without `allow_large=true` is refused.

`sophon-system-check --allocate-mib N` can probe an allocation. The trainer does not run that probe itself.

## OOM

CUDA OOM is caught and the GPU cache is cleared. The default error lists knobs: `batch_size=1`, raise `gradient_accumulation_steps`, lower `max_seq_length`, lower `max_examples`, keep `load_in_4bit=true`.

`oom_backoff=true` retries **once** with `per_device_train_batch_size=1` and accum multiplied by the old batch so the effective batch stays similar. A second OOM fails with the same hint.
