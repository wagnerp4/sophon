from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import transformers.modeling_utils as modeling_utils
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig


DEFAULT_LOCAL_DIR = "models/google-gemma-4-31b-it"


_orig_caching_allocator_warmup = modeling_utils.caching_allocator_warmup
_mps_warmup_patch_installed = False


def _device_map_targets_mps(expanded_device_map: dict) -> bool:
    for value in expanded_device_map.values():
        try:
            if torch.device(value).type == "mps":
                return True
        except (TypeError, RuntimeError, ValueError):
            continue
    return False


def _caching_allocator_warmup_lmwrap(model: object, expanded_device_map: dict, hf_quantizer: object) -> None:
    if _device_map_targets_mps(expanded_device_map):
        return
    _orig_caching_allocator_warmup(model, expanded_device_map, hf_quantizer)


def install_mps_allocator_warmup_shim() -> None:
    global _mps_warmup_patch_installed
    if _mps_warmup_patch_installed:
        return
    modeling_utils.caching_allocator_warmup = _caching_allocator_warmup_lmwrap
    _mps_warmup_patch_installed = True


def resolve_local_model_dir(model_arg: str | None) -> Path:
    if model_arg:
        return Path(model_arg).expanduser().resolve()
    env_path = os.environ.get("GEMMA4_MODEL")
    if env_path:
        return Path(env_path).expanduser().resolve()
    base = os.environ.get("GEMMA4_LOCAL_DIR", DEFAULT_LOCAL_DIR)
    return Path(base).expanduser().resolve()


def require_model_on_disk(model_dir: Path) -> str:
    if not (model_dir / "config.json").is_file():
        raise ValueError(
            "Expected a local model directory on disk with config.json at "
            f"{model_dir}. Set --model, GEMMA4_MODEL, or GEMMA4_LOCAL_DIR."
        )
    return str(model_dir)


def mps_ready() -> bool:
    return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())


def infer_default_quantization() -> str:
    raw = os.environ.get("GEMMA4_QUANTIZATION")
    if raw in ("none", "4bit", "8bit"):
        return raw
    if mps_ready():
        return "none"
    return "4bit"


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


def _make_ui_tqdm_hook(on_load_progress: Callable[[int, int, str], None]) -> Callable[..., Any]:
    from transformers.utils.logging import EmptyTqdm

    def hook(factory: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kw = dict(kwargs)
        kw["disable"] = True
        inner = factory(*args, **kw)
        if isinstance(inner, EmptyTqdm):
            return inner
        desc = str(kw.get("desc", "") or "")

        class _Bridge:
            __slots__ = ("_inner",)

            def __init__(self, wrapped: Any) -> None:
                object.__setattr__(self, "_inner", wrapped)

            def __iter__(self) -> Any:
                inner = self._inner
                total = getattr(inner, "total", None)
                if total is not None:
                    on_load_progress(0, int(total), desc)
                for item in inner:
                    n = int(getattr(inner, "n", 0))
                    tot = inner.total
                    if tot is not None:
                        on_load_progress(n, int(tot), desc)
                    yield item
                if total is not None:
                    on_load_progress(int(total), int(total), desc)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        return _Bridge(inner)

    return hook


def load_processor_and_model(
    model_path: str,
    quantization: str,
    *,
    on_load_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[AutoProcessor, AutoModelForCausalLM]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    install_mps_allocator_warmup_shim()
    from transformers.utils.logging import enable_progress_bar, set_tqdm_hook

    enable_progress_bar()
    prev_tqdm_hook = None
    if on_load_progress is not None:
        prev_tqdm_hook = set_tqdm_hook(_make_ui_tqdm_hook(on_load_progress))
    try:
        bnb = bitsandbytes_config(quantization)
        model_kw: dict[str, object] = {
            "low_cpu_mem_usage": True,
        }
        if bnb is not None:
            if not torch.cuda.is_available():
                raise ValueError(
                    "4bit and 8bit loading relies on CUDA bitsandbytes. "
                    "On Apple Silicon use quantization none so the checkpoint loads on MPS."
                )
            model_kw["device_map"] = "auto"
            model_kw["quantization_config"] = bnb
        else:
            model_kw["dtype"] = "auto"
            if mps_ready():
                model_kw["device_map"] = {"": "mps"}
            else:
                model_kw["device_map"] = "auto"
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **model_kw)
        return processor, model
    finally:
        if on_load_progress is not None:
            set_tqdm_hook(prev_tqdm_hook)


def generate_response(
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    messages: list[dict[str, object]],
    max_new_tokens: int,
    enable_thinking: bool,
) -> object:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = int(inputs["input_ids"].shape[-1])
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)


def parsed_to_display_text(parsed: object) -> str:
    if isinstance(parsed, dict):
        for key in ("content", "text", "message"):
            if key in parsed:
                inner = parsed[key]
                if isinstance(inner, str):
                    return inner
                return str(inner)
        return str(parsed)
    return str(parsed)


# TODO(multimodal): Switch to AutoModelForMultimodalLM when prompting with images or video.
