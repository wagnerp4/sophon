from __future__ import annotations

from dataclasses import replace

import click

from processing.audio.speech.cli import SttCliOptions
from processing.audio.speech.cli import format_stt_backend_help, parse_stt_backend_value
from processing.audio.speech.types import SttBackendId, normalize_stt_language


class SttBackendChoice(click.ParamType):
    name = "sst_backend"

    def convert(self, value, param, ctx):
        parsed = parse_stt_backend_value(str(value))
        if parsed is None:
            raise click.BadParameter(format_stt_backend_help(), ctx=ctx, param=param)
        return parsed


def finalize_sst(
    *,
    base: SttCliOptions | None = None,
    sst_enabled: bool,
    sst_backend: SttBackendId | None,
    sst_model: str | None,
    sst_language: str | None,
    sst_device: str | None,
    sst_max_new_tokens: int | None,
) -> SttCliOptions:
    b = base or SttCliOptions.defaults_from_env()
    out = replace(b, sst_enabled=sst_enabled)
    if sst_backend is not None:
        out = replace(out, sst_backend=sst_backend)
    if sst_model is not None and str(sst_model).strip() != "":
        out = replace(out, sst_model=str(sst_model))
    if sst_language is not None:
        out = replace(out, sst_language=normalize_stt_language(str(sst_language)))
    if sst_device is not None:
        ds = str(sst_device).strip()
        out = replace(out, sst_device=ds if ds != "" else None)
    if sst_max_new_tokens is not None:
        out = replace(out, sst_max_new_tokens=max(int(sst_max_new_tokens), 1))
    return out
