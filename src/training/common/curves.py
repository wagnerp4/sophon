from __future__ import annotations

import json
from pathlib import Path

_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], *, width: int = 28) -> str:
    if not values:
        return "·" * max(width, 1)
    seq = list(values)
    if len(seq) > width:
        seq = seq[-width:]
    lo = min(seq)
    hi = max(seq)
    span = hi - lo
    if span <= 1e-9:
        mid = _SPARK[len(_SPARK) // 2]
        return mid * len(seq)
    out: list[str] = []
    last = len(_SPARK) - 1
    for value in seq:
        idx = int(round((value - lo) / span * last))
        out.append(_SPARK[max(0, min(last, idx))])
    return "".join(out)


def losses_from_metrics_jsonl(path: Path) -> list[float]:
    if not path.is_file():
        return []
    losses: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        loss = row.get("loss")
        if isinstance(loss, (int, float)):
            losses.append(float(loss))
    return losses
