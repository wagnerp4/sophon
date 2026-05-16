from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig


def bitsandbytes_config(mode: str) -> BitsAndBytesConfig | None:
    if mode == "none":
        return None
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(mode)
