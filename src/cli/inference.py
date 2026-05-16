from __future__ import annotations

import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

# Todo: remove this path bootstrap after the package uses consistent lmwrap.* imports end-to-end.

import argparse
import os

from backend.hf.backend import (
    generate_response,
    load_processor_and_model,
    parsed_to_display_text,
    read_model_meta,
)
from backend.hf.paths import (
    infer_default_quantization,
    require_model_on_disk,
    resolve_cli_quantization,
    resolve_local_model_dir,
)
from utils.env_bootstrap import load_lmwrap_dotenv


def main() -> None:
    load_lmwrap_dotenv()
    parser = argparse.ArgumentParser(description="Text inference from local Hugging Face weights.")
    parser.add_argument(
        "prompt",
        help="User message.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local directory with config.json (default: GEMMA4_MODEL else GEMMA4_LOCAL_DIR or ./models/meta-llama-Llama-2-7b-chat-hf).",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit", "lightweight"],
        default=infer_default_quantization(),
        help="Weights: none=full; 4bit/8bit=CUDA quant; lightweight=same as --qbit 4. Ignored when --qbit is set.",
    )
    parser.add_argument(
        "--qbit",
        type=int,
        choices=[0, 4, 8],
        default=None,
        help="Short form: 0=none, 4=4bit, 8=8bit. Overrides --quantization when set. Env: GEMMA4_QBIT.",
    )
    parser.add_argument(
        "--system",
        default=os.environ.get("GEMMA4_SYSTEM_PROMPT"),
        help="Optional system message (omit if unset unless GEMMA4_SYSTEM_PROMPT is set).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Generation budget for new tokens.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable built-in reasoning (thinking) mode. If omitted, GEMMA4_THINKING controls the default.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do not strip special tokens from displayed output.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (>=1 discourages repetition).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Generation seed.")
    args = parser.parse_args()
    enable_thinking = args.thinking
    if enable_thinking is None:
        enable_thinking = os.environ.get("GEMMA4_THINKING", "").lower() in ("1", "true", "yes")
    try:
        model_path = require_model_on_disk(resolve_local_model_dir(args.model))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    processor, model = load_processor_and_model(
        model_path,
        resolve_cli_quantization(qbit=args.qbit, quantization=args.quantization),
    )
    meta = read_model_meta(model_path, processor)
    messages: list[dict[str, object]] = []
    if args.system is not None and str(args.system).strip() != "":
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    result = generate_response(
        processor,
        model,
        messages,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=enable_thinking,
        temperature=args.temperature if args.temperature is not None else meta.default_temperature,
        top_p=args.top_p if args.top_p is not None else meta.default_top_p,
        top_k=args.top_k if args.top_k is not None else meta.default_top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        strip=not bool(args.raw),
        extra_specials=meta.special_tokens,
        eos_token_ids=meta.eos_token_ids or None,
    )
    print(parsed_to_display_text(result.parsed))


if __name__ == "__main__":
    main()
