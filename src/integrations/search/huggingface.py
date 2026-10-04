from __future__ import annotations

from integrations.search.http import fetch_json, url_with_query
from integrations.search.protocol import SearchSource
from integrations.search.types import SearchHit

_MODELS = "https://huggingface.co/api/models"
_DATASETS = "https://huggingface.co/api/datasets"


class HuggingfaceSource(SearchSource):
    name = "huggingface"

    def search(self, query: str, limit: int) -> list[SearchHit]:
        datasets = _hf_rows(_DATASETS, query, limit, kind="dataset")
        models = _hf_rows(_MODELS, query, limit, kind="model")
        return (datasets + models)[:limit]


def _hf_rows(endpoint: str, query: str, limit: int, *, kind: str) -> list[SearchHit]:
    payload = fetch_json(
        url_with_query(
            endpoint,
            {
                "search": query,
                "limit": limit,
            },
        ),
        source="huggingface",
    )
    rows = payload if isinstance(payload, list) else []
    hits: list[SearchHit] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id") or row.get("modelId") or "").strip()
        if not item_id:
            continue
        extras: dict[str, str] = {"id": item_id}
        pipeline = str(row.get("pipeline_tag") or kind).strip()
        snippet = pipeline
        downloads = row.get("downloads")
        if downloads is not None:
            snippet = f"{pipeline} downloads={downloads}".strip()
        prefix = "datasets/" if kind == "dataset" else ""
        hits.append(
            SearchHit(
                title=item_id,
                url=f"https://huggingface.co/{prefix}{item_id}",
                snippet=snippet,
                extras=extras,
            )
        )
    return hits


SOURCE = HuggingfaceSource()
