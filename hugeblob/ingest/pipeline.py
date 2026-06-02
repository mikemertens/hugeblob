"""Orchestrate full ingestion: Apple Books → Readwise → EPUB/PDF files → Qdrant."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from hugeblob.config import Settings
from hugeblob.models import BookMetadata, TextChunk
from hugeblob.ingest.apple_books import load_apple_books_metadata
from hugeblob.ingest.readwise import fetch_highlights
from hugeblob.ingest.epub import epub_to_chunks
from hugeblob.ingest.pdf import pdf_to_chunks


def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"ingested_files": {}, "readwise_last_sync": None, "total_chunks": 0}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, default=str))


def _build_metadata_index(books: list[BookMetadata]) -> dict[str, BookMetadata]:
    """Index Apple Books metadata by normalised title for file matching."""
    index: dict[str, BookMetadata] = {}
    for b in books:
        key = b.title.lower().strip()
        index[key] = b
        if b.file_path:
            index[Path(b.file_path).stem.lower()] = b
    return index


def ingest_readwise(
    settings: Settings,
    store: "QdrantStore",  # noqa: F821
    embedder: "Embedder",  # noqa: F821
    log: Callable[[str], None] = print,
) -> int:
    if not settings.readwise_api_key:
        log("Skipping Readwise: no READWISE_API_KEY set.")
        return 0

    settings.ensure_data_dir()
    state = _load_state(settings.state_file)

    last_sync = state.get("readwise_last_sync")
    updated_after = datetime.fromisoformat(last_sync) if last_sync else None

    log(f"Fetching Readwise highlights{' (incremental)' if updated_after else ''}...")
    _, chunks = fetch_highlights(settings.readwise_api_key, updated_after)

    if not chunks:
        log("No new highlights.")
        return 0

    log(f"Embedding {len(chunks)} highlights...")
    _upsert_chunks(chunks, store, embedder, batch_size=100)

    state["readwise_last_sync"] = datetime.now(timezone.utc).isoformat()
    state["total_chunks"] = state.get("total_chunks", 0) + len(chunks)
    _save_state(settings.state_file, state)

    log(f"Ingested {len(chunks)} highlights from Readwise.")
    return len(chunks)


def ingest_files(
    settings: Settings,
    store: "QdrantStore",  # noqa: F821
    embedder: "Embedder",  # noqa: F821
    books_dir: Path | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> int:
    directory = (books_dir or settings.books_dir).expanduser()
    if not directory.exists():
        log(f"Books directory not found: {directory}")
        return 0

    settings.ensure_data_dir()
    state = _load_state(settings.state_file)
    ingested = state.get("ingested_files", {})

    apple_books = load_apple_books_metadata()
    meta_index = _build_metadata_index(apple_books)

    files = sorted(
        [p for p in directory.rglob("*") if p.suffix.lower() in (".epub", ".pdf")]
    )
    log(f"Found {len(files)} files in {directory}")

    total_new = 0
    for path in tqdm(files, desc="Indexing books", unit="book"):
        key = str(path)
        mtime = str(path.stat().st_mtime)

        if not force and key in ingested and ingested[key].get("mtime") == mtime:
            continue

        metadata = meta_index.get(path.stem.lower())

        try:
            if path.suffix.lower() == ".epub":
                chunks = epub_to_chunks(path, metadata, settings.chunk_size, settings.chunk_overlap)
            else:
                chunks = pdf_to_chunks(path, metadata, settings.chunk_size, settings.chunk_overlap)
        except Exception as e:
            log(f"  Error extracting {path.name}: {e}")
            continue

        if not chunks:
            continue

        _upsert_chunks(chunks, store, embedder, batch_size=100)

        ingested[key] = {"mtime": mtime, "chunk_count": len(chunks), "title": chunks[0].title}
        state["total_chunks"] = state.get("total_chunks", 0) + len(chunks)
        total_new += 1

    state["ingested_files"] = ingested
    _save_state(settings.state_file, state)

    log(f"Ingested {total_new} new/changed files.")
    return total_new


def _upsert_chunks(
    chunks: list[TextChunk],
    store: "QdrantStore",  # noqa: F821
    embedder: "Embedder",  # noqa: F821
    batch_size: int = 100,
) -> None:
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vectors = embedder.embed(texts)
        store.upsert(batch, vectors)
