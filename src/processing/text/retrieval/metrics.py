from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievalQueryMetrics:
    latency_s: float
    n_hits: int
    top_k: int
    query_chars: int
    top_score: float | None = None
    mean_score: float | None = None
    min_score: float | None = None
    score_margin: float | None = None
    backend_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def scores_from_chunks(chunks: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for chunk in chunks:
        raw = chunk.get("score")
        if raw is None:
            continue
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            continue
    return out


def build_query_metrics(
    *,
    latency_s: float,
    chunks: list[dict[str, Any]],
    top_k: int,
    query_chars: int,
    backend_id: str = "",
    extras: dict[str, Any] | None = None,
) -> RetrievalQueryMetrics:
    scores = scores_from_chunks(chunks)
    top_score = max(scores) if scores else None
    mean_score = float(statistics.fmean(scores)) if scores else None
    min_score = min(scores) if scores else None
    score_margin = None
    if len(scores) >= 2:
        ordered = sorted(scores, reverse=True)
        score_margin = float(ordered[0] - ordered[1])
    return RetrievalQueryMetrics(
        latency_s=float(latency_s),
        n_hits=len(chunks),
        top_k=int(top_k),
        query_chars=int(query_chars),
        top_score=top_score,
        mean_score=mean_score,
        min_score=min_score,
        score_margin=score_margin,
        backend_id=backend_id,
        extras=dict(extras or {}),
    )


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    if p <= 0:
        return float(ordered[0])
    if p >= 100:
        return float(ordered[-1])
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[f])
    return float(ordered[f] + (ordered[c] - ordered[f]) * (k - f))


@dataclass
class RetrievalProbeSummary:
    n: int
    empty_rate: float
    latency_mean_s: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    top_score_mean: float | None
    score_margin_mean: float | None
    hits_mean: float | None

    def format_lines(self) -> list[str]:
        def fmt(v: float | None, digits: int = 3) -> str:
            if v is None:
                return "n/a"
            return f"{v:.{digits}f}"

        return [
            f"probe n={self.n} empty_rate={self.empty_rate:.0%}",
            f"latency_s mean={fmt(self.latency_mean_s)} p50={fmt(self.latency_p50_s)} p95={fmt(self.latency_p95_s)}",
            f"top_score mean={fmt(self.top_score_mean, 4)} margin_mean={fmt(self.score_margin_mean, 4)}",
            f"hits_mean={fmt(self.hits_mean, 2)}",
        ]


def summarize_probe(rows: list[RetrievalQueryMetrics]) -> RetrievalProbeSummary:
    if not rows:
        return RetrievalProbeSummary(
            n=0,
            empty_rate=0.0,
            latency_mean_s=None,
            latency_p50_s=None,
            latency_p95_s=None,
            top_score_mean=None,
            score_margin_mean=None,
            hits_mean=None,
        )
    latencies = [r.latency_s for r in rows]
    tops = [r.top_score for r in rows if r.top_score is not None]
    margins = [r.score_margin for r in rows if r.score_margin is not None]
    hits = [float(r.n_hits) for r in rows]
    empty = sum(1 for r in rows if r.n_hits <= 0)
    return RetrievalProbeSummary(
        n=len(rows),
        empty_rate=empty / len(rows),
        latency_mean_s=float(statistics.fmean(latencies)),
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        top_score_mean=float(statistics.fmean(tops)) if tops else None,
        score_margin_mean=float(statistics.fmean(margins)) if margins else None,
        hits_mean=float(statistics.fmean(hits)),
    )


def format_query_metrics(m: RetrievalQueryMetrics) -> str:
    bits = [
        f"latency={m.latency_s:.3f}s",
        f"hits={m.n_hits}/{m.top_k}",
    ]
    if m.top_score is not None:
        bits.append(f"top={m.top_score:.4f}")
    if m.mean_score is not None:
        bits.append(f"mean={m.mean_score:.4f}")
    if m.score_margin is not None:
        bits.append(f"margin={m.score_margin:.4f}")
    if m.backend_id:
        bits.append(f"backend={m.backend_id}")
    return " ".join(bits)
