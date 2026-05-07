from __future__ import annotations

import argparse
import os
import sys

from lmwrap.gemma_backend import (
    generate_response,
    infer_default_quantization,
    load_processor_and_model,
    parsed_to_display_text,
    require_model_on_disk,
    resolve_local_model_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Text inference for Gemma 4 31B Instruct from local weights.")
    parser.add_argument(
        "prompt",
        help="User message.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local directory with config.json (default: GEMMA4_MODEL else GEMMA4_LOCAL_DIR or ./models/google-gemma-4-31b-it).",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit"],
        default=infer_default_quantization(),
        help="Weights layout: none matches MPS on macOS when GEMMA4_QUANTIZATION is unset; CUDA allows 4bit/8bit.",
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
    args = parser.parse_args()
    enable_thinking = args.thinking
    if enable_thinking is None:
        enable_thinking = os.environ.get("GEMMA4_THINKING", "").lower() in ("1", "true", "yes")
    try:
        model_path = require_model_on_disk(resolve_local_model_dir(args.model))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    processor, model = load_processor_and_model(model_path, args.quantization)
    messages: list[dict[str, object]] = []
    if args.system is not None and str(args.system).strip() != "":
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    parsed = generate_response(
        processor,
        model,
        messages,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=enable_thinking,
    )
    print(parsed_to_display_text(parsed))


if __name__ == "__main__":
    main()
