from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..protocols import RagRetriever
from ..types import RetrievalQuery, RetrievalResult, empty_result

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_BM25_K1 = 1.5
_BM25_B = 0.75


def passages_path_for_index(index_path: str | Path) -> Path:
    path = Path(index_path)
    name = path.name
    if name.endswith(".leann"):
        stem = name[: -len(".leann")]
    else:
        stem = name
    return path.parent / f"{stem}.passages.jsonl"


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


class PassageJsonlRetriever(RagRetriever):
    def __init__(self, passages_path: str | Path, *, default_top_k: int = 5) -> None:
        self._path = Path(passages_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"passage index missing: {self._path}")
        self._default_top_k = default_top_k
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._doc_tfs: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0
        self._load()

    def _load(self) -> None:
        n_tokens = 0
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                text = str(rec.get("text") or "")
                pid = str(rec.get("id") or "")
                meta = rec.get("metadata")
                tokens = tokenize(text)
                tf = Counter(tokens)
                self._ids.append(pid)
                self._texts.append(text)
                self._metadata.append(dict(meta) if isinstance(meta, dict) else {})
                self._doc_tfs.append(tf)
                self._doc_len.append(len(tokens))
                n_tokens += len(tokens)
                for term in tf:
                    self._df[term] += 1
        n_docs = len(self._ids)
        self._avgdl = (n_tokens / n_docs) if n_docs else 0.0

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        from processing.text.retrieval.metrics import build_query_metrics

        q_tokens = tokenize(query.text or "")
        if not q_tokens or not self._ids:
            return empty_result()
        params = dict(query.params)
        top_k = int(params.pop("top_k", self._default_top_k))
        if top_k <= 0:
            top_k = self._default_top_k
        n_docs = len(self._ids)
        q_unique = list(dict.fromkeys(q_tokens))
        idf: dict[str, float] = {}
        for term in q_unique:
            df = self._df.get(term, 0)
            idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        t0 = time.perf_counter()
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._doc_tfs):
            dl = self._doc_len[i]
            score = 0.0
            for term in q_unique:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                denom = freq + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (dl / self._avgdl if self._avgdl else 1.0))
                score += idf[term] * (freq * (_BM25_K1 + 1.0)) / denom
            if score > 0.0:
                scored.append((score, i))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = scored[:top_k]
        latency_s = time.perf_counter() - t0
        chunks: list[dict[str, Any]] = []
        for score, i in hits:
            chunks.append(
                {
                    "id": self._ids[i],
                    "score": float(score),
                    "text": self._texts[i],
                    "metadata": self._metadata[i],
                }
            )
        metrics = build_query_metrics(
            latency_s=latency_s,
            chunks=chunks,
            top_k=top_k,
            query_chars=len(query.text or ""),
            backend_id=self.backend_id(),
            extras={"n_passages": n_docs, "passages_path": str(self._path)},
        )
        return RetrievalResult(chunks=chunks, extras={"metrics": metrics.as_dict()})

    def backend_id(self) -> str:
        return "leann-passages"
