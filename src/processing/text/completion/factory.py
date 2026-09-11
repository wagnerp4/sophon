from __future__ import annotations

from typing import Literal

from .backends.prefix import PrefixCompleter
from .protocols import Completer

CompleterId = Literal["prefix"]


def completer_ids() -> tuple[str, ...]:
    return ("prefix",)


def load_completer(backend_id: CompleterId | str = "prefix") -> Completer:
    # TODO: fim backend (Ollama / LM Studio coder infill from CompletionQuery prefix+suffix)
    # TODO: jedi / LSP validity filter
    # TODO: hybrid ranker mixing buffer, project index, and FIM candidates
    if backend_id == "prefix":
        return PrefixCompleter()
    raise KeyError(f"unknown completion backend {backend_id!r}")
