from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from ..protocols import RagIndexer


def _load_leann_builder() -> Any:
    try:
        from leann import LeannBuilder
    except ImportError as exc:
        raise ImportError(
            "LEANN indexer requires the `leann` distribution (extras: sophon[rag-leann]). "
            "Native backends have no Windows wheels, so the chat .venv on Windows cannot "
            "install them. Create a Linux venv in WSL (not on /mnt/c) and install leann "
            "there, then run sophon-rag-index / /rag-index with that interpreter. "
            "In bash quote the extra: uv pip install --python .venv/bin/python \"leann>=0.3.7\"."
        ) from exc
    return LeannBuilder


def _as_doc_list(documents: Mapping[str, Any] | list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(documents, Mapping):
        return [documents]
    return list(documents)


class LeannIndexer(RagIndexer):
    """Build a LEANN index via LeannBuilder.add_text / build_index."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        backend_name: str = "hnsw",
        embedding_model: str | None = None,
        builder_kwargs: dict | None = None,
    ) -> None:
        self._index_path = str(Path(index_path).expanduser())
        self._backend_name = backend_name
        self._embedding_model = embedding_model
        self._builder_kwargs = dict(builder_kwargs or {})
        self._pending: list[dict[str, Any]] = []

    def backend_id(self) -> str:
        return "leann-indexer"

    def add_documents(self, documents: Mapping[str, Any] | list[Mapping[str, Any]]) -> None:
        for doc in _as_doc_list(documents):
            text = str(doc.get("text") or "").strip()
            if not text:
                continue
            metadata = dict(doc.get("metadata") or {})
            doc_id = doc.get("id")
            if doc_id is not None and "id" not in metadata:
                metadata["id"] = str(doc_id)
            self._pending.append({"text": text, "metadata": metadata})

    def clear_pending(self) -> None:
        self._pending.clear()

    def build(self, *, rebuild: bool = False) -> str:
        if not self._pending:
            raise ValueError("no documents queued for LEANN index build")
        path = Path(self._index_path)
        if rebuild:
            self._wipe_index_files(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        LeannBuilder = _load_leann_builder()
        kwargs = dict(self._builder_kwargs)
        kwargs.setdefault("backend_name", self._backend_name)
        if self._embedding_model:
            kwargs["embedding_model"] = self._embedding_model
        builder = LeannBuilder(**kwargs)
        for item in self._pending:
            builder.add_text(item["text"], metadata=item["metadata"])
        builder.build_index(self._index_path)
        n = len(self._pending)
        self._pending.clear()
        return f"built leann index path={self._index_path} chunks={n}"

    @staticmethod
    def _wipe_index_files(index_path: Path) -> None:
        parent = index_path.parent
        stem = index_path.name
        if stem.endswith(".leann"):
            stem = stem[: -len(".leann")]
        if not parent.is_dir():
            return
        for child in parent.iterdir():
            name = child.name
            if name == index_path.name or name.startswith(f"{stem}."):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass
