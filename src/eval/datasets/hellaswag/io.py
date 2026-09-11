from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"jsonl source not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no} of {path} is not a JSON object")
            yield row
