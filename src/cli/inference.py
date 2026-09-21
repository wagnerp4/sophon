from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

# Todo: remove this path bootstrap after the package uses consistent sophon.* imports end-to-end.

from backend.hf.backend import (
    generate_response,
    load_processor_and_model,
    parsed_to_display_text,
    read_model_meta,
)
from backend.hf.paths import (
    require_model_on_disk,
    resolve_cli_quantization,
    resolve_local_model_dir,
)
from processing.audio.speech.cli import TtsCliOptions, play_tts_from_options


@dataclass
class InferCliParams:
    prompt: str
    model: str | None
    quantization: str
    qbit: int | None
    system: str | None
    max_new_tokens: int
    thinking: bool | None
    raw: bool
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    seed: int | None
    tts: TtsCliOptions


def thinking_resolved(cli_flag: bool | None) -> bool:
    if cli_flag is not None:
        return cli_flag
    return os.environ.get("GEMMA4_THINKING", "").lower() in ("1", "true", "yes")


def run_infer(params: InferCliParams) -> None:
    enable_thinking = thinking_resolved(params.thinking)
    try:
        model_path = require_model_on_disk(resolve_local_model_dir(params.model))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    processor, model = load_processor_and_model(
        model_path,
        resolve_cli_quantization(qbit=params.qbit, quantization=params.quantization),
    )
    meta = read_model_meta(model_path, processor)
    messages: list[dict[str, object]] = []
    if params.system is not None and str(params.system).strip() != "":
        messages.append({"role": "system", "content": params.system})
    messages.append({"role": "user", "content": params.prompt})
    result = generate_response(
        processor,
        model,
        messages,
        max_new_tokens=params.max_new_tokens,
        enable_thinking=enable_thinking,
        temperature=params.temperature if params.temperature is not None else meta.default_temperature,
        top_p=params.top_p if params.top_p is not None else meta.default_top_p,
        top_k=params.top_k if params.top_k is not None else meta.default_top_k,
        repetition_penalty=params.repetition_penalty,
        seed=params.seed,
        strip=not bool(params.raw),
        extra_specials=meta.special_tokens,
        eos_token_ids=meta.eos_token_ids or None,
    )
    out = parsed_to_display_text(result.parsed)
    print(out)
    play_tts_from_options(params.tts, out, lambda m: print(m, file=sys.stderr, flush=True))


def main() -> None:
    import sys as _sys

    from cli.terminal import infer_command as _infer

    _infer.main(args=_sys.argv[1:], prog_name="sophon-infer", standalone_mode=True)


if __name__ == "__main__":
    main()
