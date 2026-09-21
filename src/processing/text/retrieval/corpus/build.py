from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from utils.device.env_bootstrap import sophon_data_dir, sophon_project_root

from ..backends.leann_indexer import LeannIndexer
from .chunk import documents_from_files
from .sources import collect_corpus_files, read_vault_path_from_env


ProgressFn = Callable[[str], None]


def default_leann_index_path() -> Path:
    env = os.environ.get("SOPHON_LEANN_INDEX", "").strip()
    if env:
        return Path(env).expanduser()
    return sophon_data_dir() / "rag" / "indexes" / "default"


def _index_stem(index_path: Path) -> str:
    name = index_path.name
    if name.endswith(".leann"):
        return name[: -len(".leann")]
    return name


def default_index_meta_path(index_path: Path | None = None) -> Path:
    path = index_path if index_path is not None else default_leann_index_path()
    return path.parent / f"{_index_stem(path)}.meta.json"


def default_passages_path(index_path: Path | None = None) -> Path:
    path = index_path if index_path is not None else default_leann_index_path()
    return path.parent / f"{_index_stem(path)}.passages.jsonl"


def default_index_exists(index_path: Path | None = None) -> bool:
    path = index_path if index_path is not None else default_leann_index_path()
    meta = default_index_meta_path(path)
    if meta.is_file():
        return True
    if path.is_file():
        return True
    if path.is_dir() and any(path.iterdir()):
        return True
    sibling = path.parent / f"{path.name}.meta.json"
    return sibling.is_file()


@dataclass
class CorpusBuildResult:
    index_path: str
    n_files: int
    n_chunks: int
    vault_root: str | None
    project_root: str | None
    structure_dir: str | None = None
    messages: list[str] = field(default_factory=list)


def _emit(on_progress: ProgressFn | None, msg: str) -> None:
    if on_progress is not None:
        on_progress(msg)


def build_default_corpus(
    *,
    vault_root: str | Path | None = None,
    project_root: str | Path | None = None,
    include_project: bool = True,
    index_path: str | Path | None = None,
    structure_dir: str | Path | None = None,
    rebuild: bool = False,
    chunk_chars: int = 1200,
    overlap: int = 150,
    embedding_model: str | None = None,
    on_progress: ProgressFn | None = None,
) -> CorpusBuildResult:
    vault: Path | None
    if vault_root is not None:
        vault = Path(vault_root).expanduser()
    else:
        vault = read_vault_path_from_env()

    project: Path | None = None
    if include_project:
        if project_root is not None:
            project = Path(project_root).expanduser()
        else:
            project = sophon_project_root()

    index = Path(index_path).expanduser() if index_path is not None else default_leann_index_path()
    struct = None
    if structure_dir is not None and str(structure_dir).strip():
        struct = Path(structure_dir).expanduser()
    else:
        raw_struct = os.environ.get("SOPHON_LIGHTRAG_DIR", "").strip()
        if raw_struct:
            struct = Path(raw_struct).expanduser()

    _emit(on_progress, "collecting corpus files...")
    files = collect_corpus_files(
        vault_root=vault,
        project_root=project,
        include_project=include_project,
    )
    if not files:
        raise ValueError(
            "no corpus files found. Set SOPHON_VAULT_PATH and/or keep --project enabled."
        )
    _emit(on_progress, f"files={len(files)} chunking...")
    docs = documents_from_files(files, chunk_chars=chunk_chars, overlap=overlap)
    if not docs:
        raise ValueError("corpus files produced zero text chunks")

    indexer = LeannIndexer(index, embedding_model=embedding_model)
    payload = [{"id": d.id, "text": d.text, "metadata": d.metadata} for d in docs]
    indexer.add_documents(payload)
    _emit(on_progress, f"building LEANN index at {index} (chunks={len(docs)})...")
    msg = indexer.build(rebuild=rebuild)
    messages = [msg]

    if struct is not None:
        _emit(on_progress, f"inserting into LightRAG dir={struct}...")
        from ..backends.lightrag_native import insert_texts_into_lightrag

        lr_msg = insert_texts_into_lightrag(
            struct,
            [d.text for d in docs],
            ids=[d.id for d in docs],
        )
        messages.append(lr_msg)

    return CorpusBuildResult(
        index_path=str(index),
        n_files=len(files),
        n_chunks=len(docs),
        vault_root=str(vault) if vault is not None else None,
        project_root=str(project) if project is not None else None,
        structure_dir=str(struct) if struct is not None else None,
        messages=messages,
    )
