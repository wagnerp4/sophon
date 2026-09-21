from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from ..protocols import RagRetriever
from ..types import RetrievalQuery, RetrievalResult

_LeanSearcher: Any = None
_LeanSearchImportError: Exception | None = None


def leann_searcher_available() -> bool:
    if _LeanSearcher is not None:
        return True
    if _LeanSearchImportError is not None:
        return False
    try:
        _load_leann_searcher_cls()
    except ImportError:
        return False
    return True


def _load_leann_searcher_cls() -> Any:
    global _LeanSearcher, _LeanSearchImportError
    if _LeanSearcher is not None:
        return _LeanSearcher
    if _LeanSearchImportError is not None:
        raise _LeanSearchImportError
    try:
        from leann import LeannSearcher as LS
    except ImportError as exc:
        err = ImportError(
            "leann is not importable in this interpreter. "
            "Windows TUI .venv has no LEANN wheels. Query uses passages.jsonl BM25 instead."
        )
        _LeanSearchImportError = err
        raise err from exc
    _LeanSearcher = LS
    return LS


def read_default_top_k_from_env() -> int:
    top_k_raw = os.environ.get("SOPHON_LEANN_TOP_K", "5").strip()
    try:
        return int(top_k_raw)
    except ValueError:
        return 5


def lean_native_retriever_from_env() -> RagRetriever | None:
    """SOPHON_LEANN_INDEX -> basename path passed to LeannSearcher (expects sibling .meta.json)."""
    raw = os.environ.get("SOPHON_LEANN_INDEX", "").strip()
    if not raw:
        return None
    return LeanNativeRetriever(raw, default_top_k=read_default_top_k_from_env())


class LeanNativeRetriever(RagRetriever):

    def __init__(
        self,
        index_path: str,
        *,
        default_top_k: int = 5,
        searcher_kwargs: dict | None = None,
    ) -> None:
        LS = _load_leann_searcher_cls()
        self._default_top_k = default_top_k
        sk = dict(searcher_kwargs) if searcher_kwargs else {}
        self._searcher = LS(index_path, **sk)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        import time

        from processing.text.retrieval.metrics import build_query_metrics

        params = dict(query.params)
        top_k = int(params.pop("top_k", self._default_top_k))
        complexity = int(params.pop("complexity", 64))
        beam_width = int(params.pop("beam_width", 1))
        prune_ratio = float(params.pop("prune_ratio", 0.0))
        recompute_embeddings = params.pop("recompute_embeddings", None)
        if not isinstance(recompute_embeddings, bool):
            recompute_embeddings = None
        pruning_strategy_any = params.pop("pruning_strategy", "global")
        pruning_strategy = pruning_strategy_any if isinstance(pruning_strategy_any, str) else "global"
        expected_zmq_port = int(params.pop("expected_zmq_port", 5557))
        metadata_any = params.pop("metadata_filters", None)
        metadata_filters = metadata_any if isinstance(metadata_any, dict) else None
        batch_size = int(params.pop("batch_size", 0))
        use_grep = bool(params.pop("use_grep", False))
        gemma = float(params.pop("gemma", 1.0))
        provider_any = params.pop("provider_options", None)
        provider_options = provider_any if isinstance(provider_any, dict) else None
        leftovers = dict(params)
        reco_kw = {"recompute_embeddings": recompute_embeddings} if isinstance(recompute_embeddings, bool) else {}
        t0 = time.perf_counter()
        hits = self._searcher.search(
            query.text,
            top_k=top_k,
            complexity=complexity,
            beam_width=beam_width,
            prune_ratio=prune_ratio,
            pruning_strategy=pruning_strategy,
            expected_zmq_port=expected_zmq_port,
            metadata_filters=metadata_filters,
            batch_size=batch_size,
            use_grep=use_grep,
            gemma=gemma,
            provider_options=provider_options,
            **reco_kw,
            **leftovers,
        )
        latency_s = time.perf_counter() - t0
        chunks: list[dict[str, object]] = []
        raw_hits: list[dict[str, object]] = []
        for item in hits:
            chunks.append(
                {
                    "id": item.id,
                    "score": float(item.score),
                    "text": item.text,
                    "metadata": dict(item.metadata),
                }
            )
            raw_hits.append(asdict(item))
        metrics = build_query_metrics(
            latency_s=latency_s,
            chunks=chunks,
            top_k=top_k,
            query_chars=len(query.text or ""),
            backend_id=self.backend_id(),
            extras={
                "complexity": complexity,
                "beam_width": beam_width,
                "prune_ratio": prune_ratio,
                "pruning_strategy": pruning_strategy,
            },
        )
        return RetrievalResult(
            chunks=chunks,
            extras={
                "raw_hits": raw_hits,
                "backend_search_kwargs": leftovers,
                "metrics": metrics.as_dict(),
            },
        )

    def backend_id(self) -> str:
        return "leann-native"
