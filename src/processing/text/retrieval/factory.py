from __future__ import annotations

from typing import Literal

from .backends.leann_bridge import load_retriever_from_entrypoint, resolve_leann_retriever
from .backends.lightrag_native import LightRagNativeRetriever, lightrag_native_retriever_from_env
from .backends.noop import NoopRetriever
from .protocols import RagRetriever

RagRetrieverId = Literal["noop", "leann", "lightrag"]


def rag_retriever_ids() -> tuple[str, ...]:
    return ("noop", "leann", "lightrag")


def load_rag_retriever(
    backend_id: RagRetrieverId | str = "noop",
    *,
    entrypoint: str | None = None,
    native_index_path: str | None = None,
    structure_dir: str | None = None,
) -> RagRetriever:
    """Select a retrieval backend.

    Use entrypoint= for dotted module paths.
    Use native_index_path= or SOPHON_LEANN_INDEX for LEANN.
    Use structure_dir= or SOPHON_LIGHTRAG_DIR for LightRAG.
    """
    if entrypoint:
        return load_retriever_from_entrypoint(entrypoint.strip())
    if backend_id == "noop":
        return NoopRetriever()
    if backend_id == "leann":
        return resolve_leann_retriever(native_index_path=native_index_path)
    if backend_id == "lightrag":
        if structure_dir and str(structure_dir).strip():
            return LightRagNativeRetriever(str(structure_dir).strip())
        candidate = lightrag_native_retriever_from_env()
        if candidate is None:
            raise ValueError(
                "lightrag backend requires --rag-structure-dir PATH or SOPHON_LIGHTRAG_DIR"
            )
        return candidate
    raise KeyError(f"unknown retrieval backend {backend_id!r}")
