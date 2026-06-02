"""Search helpers: wraps embedder + store into a single call."""
from __future__ import annotations

from hugeblob.embed.embedder import Embedder
from hugeblob.models import DocSource, SearchResult
from hugeblob.store.qdrant_store import QdrantStore


def search(
    query: str,
    store: QdrantStore,
    embedder: Embedder,
    limit: int = 8,
    highlights_only: bool = False,
    fulltext_only: bool = False,
    author: str | None = None,
    title: str | None = None,
    genre: str | None = None,
    min_rating: int | None = None,
) -> list[SearchResult]:
    source_filter: DocSource | None = None
    if highlights_only:
        source_filter = DocSource.HIGHLIGHT
    elif fulltext_only:
        source_filter = DocSource.FULLTEXT

    qv = embedder.embed_one(query)
    return store.search(
        query_vector=qv,
        limit=limit,
        source_filter=source_filter,
        author_filter=author,
        title_filter=title,
        genre_filter=genre,
        min_rating=min_rating,
    )
