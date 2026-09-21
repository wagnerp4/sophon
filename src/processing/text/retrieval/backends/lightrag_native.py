from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Callable

from ..protocols import RagRetriever
from ..types import RetrievalQuery, RetrievalResult

_LightRAG: Any = None
_QueryParam: Any = None
_LightRAGImportError: Exception | None = None


def _load_lightrag() -> tuple[Any, Any]:
    global _LightRAG, _QueryParam, _LightRAGImportError
    if _LightRAG is not None and _QueryParam is not None:
        return _LightRAG, _QueryParam
    if _LightRAGImportError is not None:
        raise _LightRAGImportError
    try:
        from lightrag import LightRAG, QueryParam
    except ImportError as exc:
        err = ImportError(
            "LightRAG retriever requires the `lightrag-hku` distribution "
            "(extras: sophon[rag-lightrag])."
        )
        _LightRAGImportError = err
        raise err from exc
    _LightRAG = LightRAG
    _QueryParam = QueryParam
    return LightRAG, QueryParam


def read_lightrag_dir_from_env() -> str | None:
    raw = os.environ.get("SOPHON_LIGHTRAG_DIR", "").strip()
    return raw or None


def read_lightrag_mode_from_env() -> str:
    raw = os.environ.get("SOPHON_LIGHTRAG_MODE", "hybrid").strip().lower()
    return raw or "hybrid"


def _run_coro(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _default_llm_and_embed() -> tuple[Callable[..., Any], Any]:
    """Prefer LightRAG's Ollama helpers when present; otherwise require env override later."""
    try:
        from lightrag.llm.ollama import ollama_embed, ollama_model_complete
        from lightrag.utils import EmbeddingFunc

        embed_dim = int(os.environ.get("SOPHON_LIGHTRAG_EMBED_DIM", "768"))
        embed_model = os.environ.get("SOPHON_LIGHTRAG_EMBED_MODEL", "nomic-embed-text").strip()

        async def _embed(texts: list[str]) -> Any:
            return await ollama_embed(texts, embed_model=embed_model)

        embedding_func = EmbeddingFunc(
            embedding_dim=embed_dim,
            max_token_size=8192,
            func=_embed,
        )
        return ollama_model_complete, embedding_func
    except Exception as exc:
        raise ImportError(
            "LightRAG backend needs llm/embedding callables. Install lightrag-hku with Ollama "
            "helpers or pass llm_model_func/embedding_func to LightRagNativeRetriever."
        ) from exc


_CHUNK_SPLIT = re.compile(r"\n{2,}|-----|\n(?=\d+\.\s)")


def _context_to_chunks(context: str, *, top_k: int) -> list[dict[str, object]]:
    text = str(context or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _CHUNK_SPLIT.split(text) if p and p.strip()]
    if not parts:
        parts = [text]
    chunks: list[dict[str, object]] = []
    for i, part in enumerate(parts[: max(top_k, 1)]):
        chunks.append(
            {
                "id": f"lightrag-{i}",
                "score": float(max(top_k - i, 1)),
                "text": part,
                "metadata": {"source": "lightrag"},
            }
        )
    return chunks


class LightRagNativeRetriever(RagRetriever):

    def __init__(
        self,
        working_dir: str | Path,
        *,
        default_top_k: int = 5,
        mode: str | None = None,
        llm_model_func: Callable[..., Any] | None = None,
        embedding_func: Any | None = None,
        rag_kwargs: dict | None = None,
    ) -> None:
        LightRAG, _ = _load_lightrag()
        self._default_top_k = int(default_top_k)
        self._mode = (mode or read_lightrag_mode_from_env()).strip().lower() or "hybrid"
        path = Path(working_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self._working_dir = str(path)
        llm = llm_model_func
        embed = embedding_func
        if llm is None or embed is None:
            default_llm, default_embed = _default_llm_and_embed()
            llm = llm or default_llm
            embed = embed or default_embed
        kwargs = dict(rag_kwargs or {})
        self._rag = LightRAG(
            working_dir=self._working_dir,
            llm_model_func=llm,
            embedding_func=embed,
            **kwargs,
        )
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async def _init() -> None:
            if hasattr(self._rag, "initialize_storages"):
                await self._rag.initialize_storages()
            try:
                from lightrag.kg.shared_storage import initialize_pipeline_status

                await initialize_pipeline_status()
            except Exception:
                pass

        _run_coro(_init())
        self._initialized = True

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        import time

        from processing.text.retrieval.metrics import build_query_metrics

        _, QueryParam = _load_lightrag()
        self._ensure_initialized()
        params = dict(query.params)
        top_k = int(params.pop("top_k", self._default_top_k))
        mode = str(params.pop("mode", self._mode) or self._mode)
        only_need_context = bool(params.pop("only_need_context", True))
        t0 = time.perf_counter()

        async def _query() -> Any:
            return await self._rag.aquery(
                query.text,
                param=QueryParam(
                    mode=mode,
                    only_need_context=only_need_context,
                    top_k=top_k,
                ),
            )

        raw = _run_coro(_query())
        latency_s = time.perf_counter() - t0
        if isinstance(raw, dict):
            context = str(raw.get("content") or raw.get("context") or raw.get("response") or raw)
        else:
            context = str(raw or "")
        chunks = _context_to_chunks(context, top_k=top_k)
        metrics = build_query_metrics(
            latency_s=latency_s,
            chunks=chunks,
            top_k=top_k,
            query_chars=len(query.text or ""),
            backend_id=self.backend_id(),
            extras={"mode": mode, "working_dir": self._working_dir},
        )
        return RetrievalResult(
            chunks=chunks,
            extras={
                "raw_context": context,
                "mode": mode,
                "metrics": metrics.as_dict(),
            },
        )

    def backend_id(self) -> str:
        return "lightrag-native"


def lightrag_native_retriever_from_env(*, default_top_k: int = 5) -> RagRetriever | None:
    working_dir = read_lightrag_dir_from_env()
    if not working_dir:
        return None
    return LightRagNativeRetriever(working_dir, default_top_k=default_top_k)


def insert_texts_into_lightrag(
    working_dir: str | Path,
    texts: list[str],
    *,
    ids: list[str] | None = None,
) -> str:
    """Insert plain texts into a LightRAG working directory (graph structure store)."""
    cleaned = [str(t).strip() for t in texts if str(t).strip()]
    if not cleaned:
        return "lightrag insert skipped: no texts"
    retriever = LightRagNativeRetriever(working_dir)
    retriever._ensure_initialized()
    rag = retriever._rag

    async def _insert() -> None:
        if ids is not None and len(ids) == len(cleaned) and hasattr(rag, "ainsert"):
            try:
                await rag.ainsert(cleaned, ids=ids)
                return
            except TypeError:
                pass
        if hasattr(rag, "ainsert"):
            await rag.ainsert(cleaned)
            return
        if hasattr(rag, "insert"):
            rag.insert(cleaned)
            return
        raise RuntimeError("LightRAG instance has no insert/ainsert method")

    _run_coro(_insert())
    return f"lightrag inserted texts={len(cleaned)} dir={working_dir}"
