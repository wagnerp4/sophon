from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .sources import CorpusFile


@dataclass(frozen=True)
class ChunkDoc:
    id: str
    text: str
    metadata: dict[str, object]


def chunk_text(
    text: str,
    *,
    chunk_chars: int = 1200,
    overlap: int = 150,
) -> list[str]:
    body = str(text or "").replace("\r\n", "\n").strip()
    if not body:
        return []
    size = max(int(chunk_chars), 200)
    ov = max(0, min(int(overlap), size // 2))
    if len(body) <= size:
        return [body]
    parts: list[str] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + size)
        if end < len(body):
            window = body[start:end]
            break_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
            if break_at >= size // 3:
                end = start + break_at
        piece = body[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(body):
            break
        start = max(end - ov, start + 1)
    return parts


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def documents_from_files(
    files: list[CorpusFile],
    *,
    chunk_chars: int = 1200,
    overlap: int = 150,
) -> list[ChunkDoc]:
    docs: list[ChunkDoc] = []
    for item in files:
        try:
            mtime = float(item.path.stat().st_mtime)
        except OSError:
            mtime = 0.0
        text = _read_text(item.path)
        chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{item.source}:{item.rel}#{i}"
            docs.append(
                ChunkDoc(
                    id=chunk_id,
                    text=chunk,
                    metadata={
                        "id": chunk_id,
                        "source": item.source,
                        "path": item.rel,
                        "mtime": mtime,
                    },
                )
            )
    return docs
