# Trainer

LoRA SFT for local Hugging Face causal LLMs. Chat and CLI write adapters under `data/training/adapters/`. They do not merge into the base preset.

Commands:

- `/finetune [DATASET] [on PRESET] [from PATH] [KEY=VAL ...]`
- `/finetune-config` prints the resolved recipe and the YAML path to open in the editor
- `/finetune-datasets` lists train loaders
- `/finetune-status [NAME]` reads `run_meta.json` and `metrics.jsonl` from disk
- `/adapter show | list | load NAME|PATH | clear`
- `sophon-finetune --preset KEY --dataset ID`
- `sophon-finetune --recipe config/finetune/qa-chain.yaml`
- `sophon-cli finetune --recipe PATH`

YAML lives in [`config/finetune/`](../../config/finetune/default.yaml). Load order is default → dataset overlay → preset overlay → `from PATH` → `KEY=VAL`. Unknown keys error.

This trainer is HF + PEFT + TRL only. OpenAI, Anthropic, Google, Ollama, and LM Studio are not finetune backends.

Pages:

- [models.md](models.md) availability
- [hyperparameters.md](hyperparameters.md) recipe fields
- [datasets.md](datasets.md) SFT vs `/eval model`
- [stages.md](stages.md) YAML pipeline and `continue_from`
- [devices.md](devices.md) CUDA, RAM, OOM
- [adapters.md](adapters.md) names, index, load
