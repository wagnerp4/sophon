from __future__ import annotations

from integrations.search.protocol import SearchSource

_SOURCES: dict[str, SearchSource] = {}
_LOADED = False


def register(source: SearchSource) -> None:
    _SOURCES[source.name] = source


def names() -> tuple[str, ...]:
    ensure_loaded()
    return tuple(sorted(_SOURCES))


def get(name: str) -> SearchSource:
    ensure_loaded()
    key = name.strip().lower()
    source = _SOURCES.get(key)
    if source is None:
        legal = ", ".join(("auto", *names()))
        raise ValueError(f"Unknown search source {name!r}. Legal names: {legal}")
    return source


def ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    from integrations.search import (
        arxiv,
        crossref,
        cse,
        ddg,
        europepmc,
        github,
        hn,
        huggingface,
        mdn,
        openalex,
        pubmed,
        searxng,
        semanticscholar,
        stackexchange,
        wikidata,
        wikipedia,
        zenodo,
    )

    for module in (
        arxiv,
        crossref,
        cse,
        ddg,
        europepmc,
        github,
        hn,
        huggingface,
        mdn,
        openalex,
        pubmed,
        searxng,
        semanticscholar,
        stackexchange,
        wikidata,
        wikipedia,
        zenodo,
    ):
        register(module.SOURCE)
    _LOADED = True
