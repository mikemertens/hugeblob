"""Extract plain text from PDF files using PyMuPDF."""
from __future__ import annotations

import hashlib
from pathlib import Path

from hugeblob.models import BookMetadata, TextChunk, DocSource
from hugeblob.ingest.chunker import chunk_text


def _book_id(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:16]


def pdf_to_chunks(
    path: Path,
    metadata: BookMetadata | None = None,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[TextChunk]:
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))

    pages: list[tuple[int, str]] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append((page_num, text))
    doc.close()

    if not pages:
        return []

    title = (metadata and metadata.title) or path.stem
    author = (metadata and metadata.author) or ""
    genre = (metadata and metadata.genre) or ""
    rating = metadata.rating if metadata else None
    book_id = (metadata and metadata.book_id) or _book_id(path)

    chunks: list[TextChunk] = []
    chunk_index = 0

    for page_num, page_text in pages:
        for chunk in chunk_text(page_text, chunk_size, overlap):
            chunks.append(
                TextChunk(
                    book_id=book_id,
                    title=title,
                    author=author,
                    genre=genre,
                    rating=rating,
                    source=DocSource.FULLTEXT,
                    text=chunk,
                    chunk_index=chunk_index,
                    page_number=page_num,
                    file_path=str(path),
                )
            )
            chunk_index += 1

    return chunks
