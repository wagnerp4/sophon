<div align="center">

# lmwrap

[![Python](https://img.shields.io/badge/python-3.10+-7f9a97?style=flat-square&logo=python&logoColor=white&labelColor=444444)](./pyproject.toml)
[![Model](https://img.shields.io/badge/model-Gemma%204%2031B-000000?style=flat-square&logo=huggingface&logoColor=white&labelColor=444444)](https://huggingface.co/google/gemma-4-31B-it)
[![uv](https://img.shields.io/badge/tooling-uv-7f9a97?style=flat-square&labelColor=444444)](https://docs.astral.sh/uv/)
[![Transformers](https://img.shields.io/badge/stack-Transformers-000000?style=flat-square&logo=huggingface&logoColor=white&labelColor=444444)](https://github.com/huggingface/transformers)

Small utilities for **[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)** locally using [uv](https://docs.astral.sh/uv/). Hub download, inference CLI, terminal chat (optional retrieval and SQLite memory), and benchmarks. All inference paths assume **weights are already on disk** unless you explicitly download via the bundled CLIs.

</div>

## Requirements

- **Python** 3.10+ ([`pyproject.toml`](pyproject.toml)).
- **Pillow**, **Torch**, **Torchvision**, **Transformers**, **accelerate**, **bitsandbytes** as declared in [`pyproject.toml`](pyproject.toml).
- Hugging Face access for gated checkpoints (accept terms on the model card, [`HF_TOKEN`](https://huggingface.co/docs/hub/security-tokens) or [`hf auth login`](https://huggingface.co/docs/huggingface_hub/guides/cli)).
- Enough **disk/RAM/GPU memory** for the checkpoint and quantization mode you pick.

Optional: **LEANN** corpus retrieval installs with `pip install -e ".[rag-leann]"` (see docs).

## Documentation

**Full CLI flags, retrieval, memory, env vars, modules, recipes:** [**`docs/README.md`**](docs/README.md).

Benchmark manifest schema: **`data/benchmarks/README.md`**.

## References

- [Google Gemma 4 model card (`google/gemma-4-31B-it`)](https://huggingface.co/google/gemma-4-31B-it)
- [uv](https://docs.astral.sh/uv/)
- [Transformers](https://github.com/huggingface/transformers)
- [Accelerate (`device_map`)](https://huggingface.co/docs/accelerate)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (CUDA 4-/8-bit loading)

## License

Upstream Gemma weights and terms follow the Hugging Face model card and licensors. This repository ships tooling only.
