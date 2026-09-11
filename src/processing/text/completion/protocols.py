from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import CompletionCandidate, CompletionQuery


@runtime_checkable
class Completer(Protocol):
    def complete(self, query: CompletionQuery) -> CompletionCandidate | None: ...

    def backend_id(self) -> str: ...
