<div align="center">

# mithril

[![Python](https://img.shields.io/badge/python-3.10+-7f9a97?style=flat-square&logo=python&logoColor=white&labelColor=444444)](./pyproject.toml)
[![Model](https://img.shields.io/badge/model-Gemma%204%2031B-000000?style=flat-square&logo=huggingface&logoColor=white&labelColor=444444)](https://huggingface.co/google/gemma-4-31B-it)
[![uv](https://img.shields.io/badge/tooling-uv-7f9a97?style=flat-square&labelColor=444444)](https://docs.astral.sh/uv/)
[![Transformers](https://img.shields.io/badge/stack-Transformers-000000?style=flat-square&logo=huggingface&logoColor=white&labelColor=444444)](https://github.com/huggingface/transformers)


abc

</div>

## Requirements

- **Python** 3.10+ ([`pyproject.toml`](pyproject.toml)).
- **Pillow**, **Torch**, **Torchvision**, **Transformers**, **accelerate**, **bitsandbytes** as declared in [`pyproject.toml`](pyproject.toml).
- Hugging Face access for gated checkpoints (accept terms on the model card, [`HF_TOKEN`](https://huggingface.co/docs/hub/security-tokens) or [`hf auth login`](https://huggingface.co/docs/huggingface_hub/guides/cli)).
- Enough **disk/RAM/GPU memory** for the checkpoint and quantization mode you pick.

## Run

```bash
    # mithril Textual home + chat TUI (default with --tui)
    mithril-cli chat --tui --interface textual --preset llama2_7b_chat

    # classic REPL in current terminal
    mithril-cli chat --interface repl --preset llama2_7b_chat

    # external Toad hub (loose coupling)
    mithril-cli chat --interface toad
```