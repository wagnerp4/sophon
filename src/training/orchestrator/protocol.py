from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from training.orchestrator.spec import TrainerSpec

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class CatalogRow:
    kind: str
    key: str
    status: str
    detail: str
    size_hint: str = ""


class Subtrainer(Protocol):
    driver_id: str

    def inventory(self) -> list[CatalogRow]: ...

    def preflight(self, spec: TrainerSpec) -> list[str]: ...

    def fetch_data(self, key: str, on_log: LogFn) -> None: ...

    def run(self, spec: TrainerSpec, on_log: LogFn) -> None: ...

    def status(self) -> list[str]: ...

    def stop(self) -> list[str]: ...
