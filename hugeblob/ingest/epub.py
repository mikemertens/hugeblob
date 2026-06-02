"""Extract plain text from EPUB files."""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

from bs4 import BeautifulSoup

from hugeblob.models import BookMetadata, TextChunk, DocSource
from hugeblob.ingest.chunker import chunk_text

# ebooklib spams deprecation warnings on Python 3.12+
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ebooklib")


def _extract_text(epub_path: Path) -> tuple[str, dict]:
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})

    meta: dict = {}
    for key, value in book.metadata.get("http://purl.org/dc/elements/1.1/", {}).items():
        if value:
            meta[key] = value[0][0] if value[0] else ""

    parts: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)

    return "\n\n".join(parts), meta


def _book_id(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:16]


def epub_to_chunks(
    path: Path,
    metadata: BookMetadata | None = None,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[TextChunk]:
    full_text, meta = _extract_text(path)
    if not full_text.strip():
        return []

    title = (metadata and metadata.title) or meta.get("title", path.stem)
    author = (metadata and metadata.author) or meta.get("creator", "")
    genre = (metadata and metadata.genre) or ""
    rating = metadata.rating if metadata else None
    book_id = (metadata and metadata.book_id) or _book_id(path)

    raw_chunks = chunk_text(full_text, chunk_size, overlap)
    return [
        TextChunk(
            book_id=book_id,
            title=title,
            author=author,
            genre=genre,
            rating=rating,
            source=DocSource.FULLTEXT,
            text=chunk,
            chunk_index=i,
            file_path=str(path),
        )
        for i, chunk in enumerate(raw_chunks)
    ]
