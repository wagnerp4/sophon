from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import transformers.modeling_utils as modeling_utils
from transformers import AutoModelForCausalLM, AutoProcessor

from backend.hf.const import BUILTIN_SPECIAL_TOKEN_PATTERNS
from backend.shared import bitsandbytes_config

_MULTIMODAL_MODEL_TYPES = frozenset({"gemma4_unified"})
_IMAGE_TEXT_MODEL_TYPES = frozenset({"gemma4", "gemma4_assistant"})


_orig_caching_allocator_warmup = modeling_utils.caching_allocator_warmup
_mps_warmup_patch_installed = False


@dataclass
class GenerationResult:
    parsed: object
    input_tokens: int
    new_tokens: int
    gen_time_s: float
    stop_reason: str
    raw_response: str
    reasoning: str | None = None
    plan: str | None = None


@dataclass
class ModelMeta:
    max_position_embeddings: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    eos_token_ids: list[int] = field(default_factory=list)
    special_tokens: list[str] = field(default_factory=list)


def _device_map_targets_mps(expanded_device_map: dict) -> bool:
    for value in expanded_device_map.values():
        try:
            if torch.device(value).type == "mps":
                return True
        except (TypeError, RuntimeError, ValueError):
            continue
    return False


def _caching_allocator_warmup_sophon(model: object, expanded_device_map: dict, hf_quantizer: object) -> None:
    if _device_map_targets_mps(expanded_device_map):
        return
    _orig_caching_allocator_warmup(model, expanded_device_map, hf_quantizer)


def install_mps_allocator_warmup_shim() -> None:
    global _mps_warmup_patch_installed
    if _mps_warmup_patch_installed:
        return
    modeling_utils.caching_allocator_warmup = _caching_allocator_warmup_sophon
    _mps_warmup_patch_installed = True


def mps_ready() -> bool:
    return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())


def _cuda_quantized_max_memory(reserve_gib: float = 1.5) -> dict[int | str, str]:
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / (1024**3)
    budget_gib = max(1.0, total_gib - reserve_gib)
    return {0: f"{int(budget_gib)}GiB", "cpu": "0GiB"}


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


def _read_config_json(model_path: str) -> dict[str, Any] | None:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_model_type(model_path: str) -> str | None:
    data = _read_config_json(model_path)
    if data is None:
        return None
    model_type = data.get("model_type")
    return model_type if isinstance(model_type, str) and model_type else None


def _require_gemma4_unified_support(model_type: str | None) -> None:
    if model_type != "gemma4_unified":
        return
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "gemma4_unified" in CONFIG_MAPPING:
        return
    raise ValueError(
        "gemma4_unified requires transformers>=5.10.1. "
        "From the sophon repo run: uv sync"
    )


def _model_loader_for_type(model_type: str | None):
    if model_type in _MULTIMODAL_MODEL_TYPES:
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM.from_pretrained
    if model_type in _IMAGE_TEXT_MODEL_TYPES:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained
    return AutoModelForCausalLM.from_pretrained


def _model_uses_chat_template_inputs(model_type: str | None) -> bool:
    if model_type is None:
        return False
    return model_type in _MULTIMODAL_MODEL_TYPES or model_type in _IMAGE_TEXT_MODEL_TYPES


def _apply_chat_inputs(
    processor: AutoProcessor,
    messages: list[dict[str, object]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool,
    tools: list | None = None,
) -> dict[str, torch.Tensor]:
    template_kwargs: dict[str, object] = {
        "conversation": messages,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        template_kwargs["tools"] = tools
    try:
        inputs = processor.apply_chat_template(**template_kwargs, enable_thinking=enable_thinking)
    except TypeError:
        template_kwargs.pop("tools", None)
        try:
            inputs = processor.apply_chat_template(**template_kwargs, enable_thinking=enable_thinking)
        except TypeError:
            inputs = processor.apply_chat_template(**template_kwargs)
    if isinstance(inputs, dict):
        return inputs
    input_ids = getattr(inputs, "get", lambda _k, _d=None: None)("input_ids")
    if input_ids is None:
        raise TypeError("apply_chat_template did not return tokenized multimodal inputs.")
    return inputs


def _move_inputs_to_model(inputs: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    device = model.device
    dtype = getattr(model, "dtype", None)
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.to(device)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        moved[key] = tensor
    return moved


def _read_text_config_max_pos(model_path: str) -> int | None:
    data = _read_config_json(model_path)
    if data is None:
        return None
    candidates: list[Any] = []
    if isinstance(data, dict):
        text_cfg = data.get("text_config")
        if isinstance(text_cfg, dict) and "max_position_embeddings" in text_cfg:
            candidates.append(text_cfg["max_position_embeddings"])
        for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            if key in data:
                candidates.append(data[key])
    for v in candidates:
        if isinstance(v, int) and v > 0:
            return v
    return None


def _read_generation_defaults(model_path: str) -> tuple[float | None, float | None, int | None, list[int]]:
    gen_path = Path(model_path) / "generation_config.json"
    if not gen_path.is_file():
        return None, None, None, []
    try:
        data = json.loads(gen_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, None, []
    if not isinstance(data, dict):
        return None, None, None, []
    temp = data.get("temperature")
    tp = data.get("top_p")
    tk = data.get("top_k")
    eos = data.get("eos_token_id")
    eos_list: list[int] = []
    if isinstance(eos, int):
        eos_list = [eos]
    elif isinstance(eos, list):
        eos_list = [int(x) for x in eos if isinstance(x, int)]
    return (
        float(temp) if isinstance(temp, (int, float)) else None,
        float(tp) if isinstance(tp, (int, float)) else None,
        int(tk) if isinstance(tk, int) else None,
        eos_list,
    )


def _collect_special_tokens(processor: object) -> list[str]:
    tok = getattr(processor, "tokenizer", None) or processor
    out: list[str] = []
    for attr in ("eos_token", "bos_token", "pad_token", "unk_token"):
        v = getattr(tok, attr, None)
        if isinstance(v, str) and v:
            out.append(v)
    add = getattr(tok, "additional_special_tokens", None)
    if isinstance(add, list):
        out.extend([str(s) for s in add if isinstance(s, str) and s])
    return out


def read_model_meta(model_path: str, processor: object) -> ModelMeta:
    temp, tp, tk, eos = _read_generation_defaults(model_path)
    return ModelMeta(
        max_position_embeddings=_read_text_config_max_pos(model_path),
        default_temperature=temp,
        default_top_p=tp,
        default_top_k=tk,
        eos_token_ids=eos,
        special_tokens=_collect_special_tokens(processor),
    )


def _maybe_load_phase(
    cb: Callable[[int, int, str], None] | None,
    description: str,
) -> None:
    if cb is None:
        return
    cb(0, 0, description)


def _log_load(
    cb: Callable[[int, int, str], None] | None,
    description: str,
) -> None:
    if cb is not None:
        cb(0, 0, description)
        return
    print(f"sophon: {description}", file=sys.stderr, flush=True)


def load_processor_and_model(
    model_path: str,
    quantization: str,
    *,
    on_load_progress: Callable[[int, int, str], None] | None = None,
    adapter_path: str | None = None,
) -> tuple[AutoProcessor, nn.Module]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    install_mps_allocator_warmup_shim()
    import transformers.utils.logging as _tr_logging

    enable_pb = getattr(_tr_logging, "enable_progress_bar", None)
    set_hook = getattr(_tr_logging, "set_tqdm_hook", None)
    if callable(enable_pb):
        enable_pb()
    prev_tqdm_hook = None
    if on_load_progress is not None and callable(set_hook):
        prev_tqdm_hook = set_hook(_make_ui_tqdm_hook(on_load_progress))
    try:
        _log_load(on_load_progress, "loading processor (local files)...")
        _maybe_load_phase(on_load_progress, "Loading tokenizer and processor (disk I/O)...")
        bnb = bitsandbytes_config(quantization)
        model_kw: dict[str, object] = {
            "low_cpu_mem_usage": True,
        }
        if sys.platform == "win32":
            model_kw["disable_mmap"] = True
        if bnb is not None:
            if not torch.cuda.is_available():
                raise ValueError(
                    "4bit and 8bit loading rely on CUDA and bitsandbytes "
                    "(torch.cuda.is_available() must be True). "
                    "Use --quantization none or --qbit 0 for CPU-only or non-CUDA PyTorch. "
                    "For GPUs, install a CUDA build from https://pytorch.org/get-started/locally/"
                )
            model_kw["device_map"] = "auto"
            model_kw["quantization_config"] = bnb
            model_kw["max_memory"] = _cuda_quantized_max_memory()
        else:
            model_kw["dtype"] = "auto"
            if mps_ready():
                model_kw["device_map"] = {"": "mps"}
            else:
                model_kw["device_map"] = "auto"
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        _maybe_load_phase(
            on_load_progress,
            "Loading tensors into GPU RAM (quantization="
            + str(quantization)
            + "). Quiet stretches are normal until shard tqdm starts.",
        )
        _log_load(
            on_load_progress,
            "loading weights. progress may pause for minutes on a large shard or 4-bit init.",
        )
        model_type = _read_model_type(model_path)
        _require_gemma4_unified_support(model_type)
        model_loader = _model_loader_for_type(model_type)
        try:
            model = model_loader(model_path, local_files_only=True, **model_kw)
        except TypeError as exc:
            if sys.platform != "win32" or "disable_mmap" not in str(exc):
                raise
            model_kw.pop("disable_mmap", None)
            model = model_loader(model_path, local_files_only=True, **model_kw)
        if adapter_path:
            adapter_resolved = Path(adapter_path).expanduser().resolve()
            if not adapter_resolved.is_dir():
                raise ValueError(f"adapter directory not found: {adapter_resolved}")
            _log_load(on_load_progress, f"loading LoRA adapter from {adapter_resolved} ...")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ValueError(
                    "loading adapters requires peft. Run: uv sync --extra finetune"
                ) from exc
            model = PeftModel.from_pretrained(model, str(adapter_resolved))
        _log_load(on_load_progress, "model load finished.")
        return processor, model
    finally:
        if on_load_progress is not None and callable(set_hook):
            set_hook(prev_tqdm_hook)


def _build_strip_pattern(extra_specials: list[str] | None = None) -> re.Pattern[str]:
    parts: list[str] = list(BUILTIN_SPECIAL_TOKEN_PATTERNS)
    if extra_specials:
        for tok in extra_specials:
            if not isinstance(tok, str) or not tok:
                continue
            parts.append(re.escape(tok))
    pattern = "|".join(parts)
    return re.compile(pattern)


def strip_special_tokens(text: str, extra_specials: list[str] | None = None) -> str:
    if not text:
        return text
    pat = _build_strip_pattern(extra_specials)
    cleaned = pat.sub("", text)
    return cleaned.strip()


def count_prompt_tokens(
    processor: AutoProcessor,
    messages: list[dict[str, object]],
    *,
    enable_thinking: bool = False,
    model_type: str | None = None,
) -> int:
    try:
        if _model_uses_chat_template_inputs(model_type):
            inputs = _apply_chat_inputs(
                processor,
                messages,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
            ids = inputs.get("input_ids")
        else:
            try:
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = processor(text=text, return_tensors="pt")
            ids = inputs.get("input_ids") if isinstance(inputs, dict) else inputs["input_ids"]
        if ids is None:
            return 0
        return int(ids.shape[-1])
    except Exception:
        return 0


def _resolve_eos_ids(model: nn.Module, override: list[int] | None) -> list[int]:
    if override:
        return [int(x) for x in override]
    gc = getattr(model, "generation_config", None)
    raw = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, list):
        return [int(x) for x in raw if isinstance(x, int)]
    return []


def generate_response(
    processor: AutoProcessor,
    model: nn.Module,
    messages: list[dict[str, object]],
    max_new_tokens: int,
    enable_thinking: bool,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    seed: int | None = None,
    do_sample: bool | None = None,
    strip: bool = True,
    extra_specials: list[str] | None = None,
    eos_token_ids: list[int] | None = None,
    model_type: str | None = None,
    tools: list | None = None,
) -> GenerationResult:
    resolved_model_type = model_type or getattr(getattr(model, "config", None), "model_type", None)
    if _model_uses_chat_template_inputs(resolved_model_type):
        inputs = _move_inputs_to_model(
            _apply_chat_inputs(
                processor,
                messages,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
                tools=tools,
            ),
            model,
        )
    else:
        template_kwargs: dict[str, object] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            template_kwargs["tools"] = tools
        try:
            text = processor.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **template_kwargs,
            )
        except TypeError:
            template_kwargs.pop("tools", None)
            try:
                text = processor.apply_chat_template(
                    messages,
                    enable_thinking=enable_thinking,
                    **template_kwargs,
                )
            except TypeError:
                text = processor.apply_chat_template(messages, **template_kwargs)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    sampling_active = False
    if temperature is not None:
        gen_kwargs["temperature"] = float(temperature)
        sampling_active = True
    if top_p is not None:
        gen_kwargs["top_p"] = float(top_p)
        sampling_active = True
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)
        sampling_active = True
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)
    if do_sample is not None:
        gen_kwargs["do_sample"] = bool(do_sample)
    elif sampling_active:
        gen_kwargs["do_sample"] = True

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    t0 = time.perf_counter()
    outputs = model.generate(**inputs, **gen_kwargs)  # type: ignore[call-arg]
    gen_time_s = max(0.0, time.perf_counter() - t0)

    new_token_ids = outputs[0][input_len:]
    new_tokens = int(new_token_ids.shape[-1])

    decode_fn = getattr(processor, "decode", None)
    if decode_fn is None:
        tok = getattr(processor, "tokenizer", None)
        if tok is None:
            raise TypeError("Processor has no decode() and no .tokenizer for decoding.")
        decode_fn = tok.decode
    raw_response = decode_fn(new_token_ids, skip_special_tokens=False)

    nested_tok = getattr(processor, "tokenizer", None)
    schema_here = getattr(processor, "response_schema", None)
    if schema_here is None and nested_tok is not None:
        schema_here = getattr(nested_tok, "response_schema", None)
    parse_fn = getattr(processor, "parse_response", None)
    if callable(parse_fn) and schema_here is not None:
        parsed: object = parse_fn(raw_response)
    else:
        parsed = raw_response

    eos_ids = _resolve_eos_ids(model, eos_token_ids)
    last_tok = int(new_token_ids[-1].item()) if new_tokens > 0 else -1
    stop_reason = "eos" if (eos_ids and last_tok in eos_ids) else "max_new_tokens"

    reasoning, plan = _parsed_side_channels(parsed)
    if reasoning is None:
        reasoning = _reasoning_from_raw_response(raw_response)

    if strip:
        display = parsed_to_display_text(parsed)
        cleaned = strip_special_tokens(display, extra_specials=extra_specials)
        parsed = cleaned

    return GenerationResult(
        parsed=parsed,
        input_tokens=input_len,
        new_tokens=new_tokens,
        gen_time_s=gen_time_s,
        stop_reason=stop_reason,
        raw_response=raw_response,
        reasoning=reasoning,
        plan=plan,
    )


def _parsed_side_channels(parsed: object) -> tuple[str | None, str | None]:
    if not isinstance(parsed, dict):
        return None, None
    reasoning = None
    for key in ("thinking", "reasoning", "reasoning_content", "thought"):
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            reasoning = val.strip()
            break
    plan = None
    for key in ("plan", "planning"):
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            plan = val.strip()
            break
        if isinstance(val, list):
            parts = [str(item).strip() for item in val if str(item).strip()]
            if parts:
                plan = "\n".join(f"{i}. {item}" for i, item in enumerate(parts, 1))
                break
    return reasoning, plan


def _reasoning_from_raw_response(raw_response: str) -> str | None:
    # TODO: parse Gemma/Qwen think/channel tags from raw_response when parsed is a plain string.
    _ = raw_response
    return None


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


def hf_chat_complete(
    processor: AutoProcessor,
    model: nn.Module,
    messages: list[dict[str, object]],
    *,
    max_new_tokens: int,
    enable_thinking: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    seed: int | None = None,
    extra_specials: list[str] | None = None,
    eos_token_ids: list[int] | None = None,
    model_type: str | None = None,
    tools: list | None = None,
):
    from backend.hf.tool_parse import parse_generated_tool_calls, strip_tool_call_text
    from backend.openai_compat import ChatCompletionResult

    result = generate_response(
        processor,
        model,
        messages,
        max_new_tokens,
        enable_thinking,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
        strip=False,
        extra_specials=extra_specials,
        eos_token_ids=eos_token_ids,
        model_type=model_type,
        tools=tools,
    )
    calls = parse_generated_tool_calls(result.raw_response, result.parsed)
    text = parsed_to_display_text(result.parsed)
    if calls:
        text = strip_tool_call_text(text)
    else:
        text = strip_special_tokens(text, extra_specials=extra_specials)
    finish = "tool_calls" if calls else result.stop_reason
    return ChatCompletionResult(
        text=text,
        tool_calls=calls,
        prompt_tokens=result.input_tokens,
        completion_tokens=result.new_tokens,
        finish_reason=finish,
        reasoning=result.reasoning,
    )


