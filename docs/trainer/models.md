# Finetune model availability

Eligibility is stored on `HFModelPreset` in [`src/backend/hf/registry.py`](../../src/backend/hf/registry.py) (`finetune_eligible`, `finetune_requires_allow_large`, `finetune_reason`, `vram_class`). `/finetune` calls `require_finetune_preset`. Weights must already be on disk (`config.json` plus complete shards). Use `/model-download PRESET` first.

Train uses `AutoModelForCausalLM` and LoRA. Chat inference may use a different loader for multimodal presets. Those two paths are not the same.

## Eligible (7B-13B class)

LoRA + 4-bit is the default. CUDA is required.

- `llama2_7b_chat` — `meta-llama/Llama-2-7b-chat-hf` — `models/meta-llama-Llama-2-7b-chat-hf` — vram_class `7b`
- `llama2_13b_chat` — `meta-llama/Llama-2-13b-chat-hf` — vram_class `13b`
- `llama3_8b_instruct` — `meta-llama/Meta-Llama-3-8B-Instruct` — vram_class `8b`
- `llama3_1_8b_instruct` — `meta-llama/Llama-3.1-8B-Instruct` — vram_class `8b`
- `vicuna_7b_v1_5` — `lmsys/vicuna-7b-v1.5` — vram_class `7b`
- `vicuna_13b_v1_5` — `lmsys/vicuna-13b-v1.5` — vram_class `13b`

## Eligible but gated

- `llama3_70b_instruct` — vram_class `70b`. Refused unless `allow_large=true`. There is no Llama 3.1 70B preset in this registry.

## Ineligible

The job fails with `finetune_reason` from the registry.

- Gemma 4 family (`gemma4_*`): multimodal / image-text loaders, not causal LoRA SFT
- Llama 4 (`llama4_*`): MoE vision
- Every Qwen row currently registered: image-text, TTS, ASR, GPTQ, or FP8
- API and server backends: openai, anthropic, google, ollama, lmstudio

GPTQ and FP8 are already quantized. This trainer does not stack bitsandbytes 4-bit LoRA on top of them.
