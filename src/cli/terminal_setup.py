from __future__ import annotations

from dataclasses import replace

import click

from processing.audio.speech.cli import TtsCliOptions
from processing.audio.speech.cli import format_tts_backend_help, parse_tts_backend_value
from processing.audio.speech.types import TtsBackendId


class TtsBackendChoice(click.ParamType):
    name = "tts_backend"

    def convert(self, value, param, ctx):
        parsed = parse_tts_backend_value(str(value))
        if parsed is None:
            raise click.BadParameter(format_tts_backend_help(), ctx=ctx, param=param)
        return parsed


def finalize_tts(
    *,
    base: TtsCliOptions | None = None,
    tts_enabled: bool,
    tts_backend: TtsBackendId | None,
    tts_model: str | None,
    tts_ollama_model: str | None,
    tts_speaker: str | None,
    tts_language: str | None,
    tts_instruct: str | None,
    tts_max_chars: int | None,
    tts_device: str | None,
    tts_raw_output: bool,
) -> TtsCliOptions:
    b = base or TtsCliOptions.defaults_from_env()
    out = replace(b, tts_enabled=tts_enabled, tts_raw_output=tts_raw_output)
    if tts_backend is not None:
        out = replace(out, tts_backend=tts_backend)
    if tts_model is not None and str(tts_model).strip() != "":
        out = replace(out, tts_model=str(tts_model))
    if tts_ollama_model is not None:
        stripped = str(tts_ollama_model).strip()
        if stripped.lower() in ("clear", "none"):
            out = replace(out, tts_ollama_model=None)
        elif stripped != "":
            out = replace(out, tts_ollama_model=stripped)
    if tts_speaker is not None and str(tts_speaker).strip() != "":
        out = replace(out, tts_speaker=str(tts_speaker))
    if tts_language is not None and str(tts_language).strip() != "":
        out = replace(out, tts_language=str(tts_language))
    if tts_instruct is not None:
        stripped = str(tts_instruct).strip()
        if stripped == "" or stripped.lower() in ("clear", "none"):
            out = replace(out, tts_instruct=None)
        else:
            out = replace(out, tts_instruct=stripped)
    if tts_max_chars is not None:
        out = replace(out, tts_max_chars=max(int(tts_max_chars), 1))
    if tts_device is not None:
        ds = str(tts_device).strip()
        out = replace(out, tts_device=ds if ds != "" else None)
    return out
