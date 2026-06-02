"""Qdrant vector store wrapper."""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from hugeblob.models import TextChunk, SearchResult, DocSource


class QdrantStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "library",
        vector_size: int = 1536,
    ):
        self.collection = collection
        self.vector_size = vector_size
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def upsert(self, chunks: list[TextChunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.book_id}:{c.source}:{c.highlight_id or c.chunk_index}")),
                vector=v,
                payload=c.model_dump(),
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        source_filter: DocSource | None = None,
        author_filter: str | None = None,
        title_filter: str | None = None,
        genre_filter: str | None = None,
        min_rating: int | None = None,
    ) -> list[SearchResult]:
        must: list[Any] = []

        if source_filter:
            must.append(FieldCondition(key="source", match=MatchValue(value=source_filter.value)))
        if author_filter:
            must.append(FieldCondition(key="author", match=MatchValue(value=author_filter)))
        if title_filter:
            must.append(FieldCondition(key="title", match=MatchValue(value=title_filter)))
        if genre_filter:
            must.append(FieldCondition(key="genre", match=MatchValue(value=genre_filter)))
        if min_rating is not None:
            from qdrant_client.models import Range
            must.append(FieldCondition(key="rating", range=Range(gte=min_rating)))

        query_filter = Filter(must=must) if must else None

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for hit in hits:
            p = hit.payload or {}
            results.append(
                SearchResult(
                    score=hit.score,
                    text=p.get("text", ""),
                    title=p.get("title", ""),
                    author=p.get("author", ""),
                    source=DocSource(p.get("source", "fulltext")),
                    book_id=p.get("book_id", ""),
                    page_number=p.get("page_number"),
                    file_path=p.get("file_path"),
                    highlight_note=p.get("highlight_note"),
                    readwise_url=p.get("readwise_url"),
                )
            )
        return results

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count

    def distinct_books(self) -> list[dict]:
        """Return one payload per unique book_id (for stats/listing)."""
        seen: dict[str, dict] = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=500,
                offset=offset,
                with_payload=["book_id", "title", "author", "genre", "rating", "source"],
            )
            for r in records:
                p = r.payload or {}
                bid = p.get("book_id", "")
                if bid and bid not in seen:
                    seen[bid] = p
            if offset is None:
                break
        return list(seen.values())


def store_from_settings(settings) -> QdrantStore:
    return QdrantStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
        vector_size=settings.embedding_dim,
    )
