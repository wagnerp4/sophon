from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from ..protocols import RagRetriever

from .callable_backend import CallableRetriever
from .noop import NoopRetriever

_LEANN_ENV_ENTRY = "SOPHON_LEANN_ENTRYPOINT"
_LEANN_ENV_ROOT = "SOPHON_LEANN_ROOT"


def _normalize_entrypoint(ep: str) -> tuple[str, str | None]:
    if ":" not in ep:
        raise ValueError(f"SOPHON_LEANN_ENTRYPOINT must be module:function, got {ep!r}")
    mod, _, attr = ep.partition(":")
    mod = mod.strip()
    tail = attr.strip()
    fn = tail if tail else None
    return mod, fn


def _resolve_attribute(module: Any, dotted: str | None) -> Any:
    if dotted is None or dotted == "":
        return module
    cur = module
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def _ensure_leann_on_path(extra_root: str | None) -> None:
    if not extra_root:
        return
    root = Path(extra_root).expanduser().resolve()
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_retriever_from_entrypoint(entrypoint: str) -> RagRetriever:
    """Import module:function and wrap as a CallableRetriever (backend_id='leann')."""
    module_name, fn_path = _normalize_entrypoint(entrypoint)
    module = importlib.import_module(module_name)
    resolved = _resolve_attribute(module, fn_path)
    if not callable(resolved):
        raise TypeError(f"LEANN entrypoint {entrypoint!r} did not resolve to a callable")
    return CallableRetriever(resolved, backend_id="leann")


def resolve_leann_retriever(*, native_index_path: str | None = None) -> RagRetriever:
    """
    Opt-in LEANN resolver.
    SOPHON_LEANN_ENTRYPOINT binds a custom callable.
    SOPHON_LEANN_INDEX or native_index_path binds leann.LeannSearcher directly.
    SOPHON_LEANN_ROOT prepends a LEANN checkout for editable installs without packaging.
    """
    _ensure_leann_on_path(os.environ.get(_LEANN_ENV_ROOT))
    ep = os.environ.get(_LEANN_ENV_ENTRY, "").strip()
    if ep:
        return load_retriever_from_entrypoint(ep)
    resolved_path = ""
    if native_index_path:
        resolved_path = native_index_path.strip()
    if not resolved_path:
        resolved_path = os.environ.get("SOPHON_LEANN_INDEX", "").strip()
    if resolved_path:
        from .leann_native import LeanNativeRetriever, leann_searcher_available, read_default_top_k_from_env

        if leann_searcher_available():
            return LeanNativeRetriever(resolved_path, default_top_k=read_default_top_k_from_env())
        from .passages_jsonl import PassageJsonlRetriever, passages_path_for_index

        passages = passages_path_for_index(resolved_path)
        # TODO: WSL sidecar to Linux LeannSearcher for dense DiskANN from the Windows TUI
        if passages.is_file():
            return PassageJsonlRetriever(
                passages, default_top_k=read_default_top_k_from_env()
            )
        raise ImportError(
            "leann is not importable and no passages.jsonl sibling exists at "
            f"{passages}. Rebuild the corpus from WSL, then retry."
        )
    return NoopRetriever()
