"""Extract book metadata from the macOS Apple Books library."""
from __future__ import annotations

import hashlib
import plistlib
from pathlib import Path
from typing import Any

from hugeblob.models import BookMetadata

# Apple Books has moved its plist across macOS versions — try all known locations.
_PLIST_CANDIDATES = [
    Path.home() / "Library/Containers/com.apple.BKAgentService/Data/Documents/iBooks/Books/Books.plist",
    Path.home() / "Library/Group Containers/group.com.apple.iBooks/Documents/iBooks/Books/Books.plist",
    Path.home() / "Library/Containers/com.apple.iBooksX/Data/Documents/iBooks/Books/Books.plist",
]

# Apple Books iCloud storage root
_ICLOUD_BOOKS = Path.home() / "Library/Mobile Documents/iCloud~com~apple~iBooks/Documents"


def _book_id(title: str, author: str) -> str:
    return hashlib.md5(f"{title}|{author}".encode()).hexdigest()[:16]


def _parse_entry(entry: dict[str, Any]) -> BookMetadata | None:
    title = (
        entry.get("BKBookTitle")
        or entry.get("title")
        or entry.get("itemName")
        or ""
    ).strip()
    if not title:
        return None

    author = (
        entry.get("BKBookAuthor")
        or entry.get("artistName")
        or entry.get("author")
        or ""
    ).strip()

    genre = (
        entry.get("BKBookGenre")
        or entry.get("primaryGenreName")
        or entry.get("genre")
        or ""
    ).strip()

    description = (
        entry.get("BKBookDescription")
        or entry.get("description")
        or ""
    ).strip()

    rating_raw = entry.get("BKBookRating") or entry.get("rating")
    rating: int | None = None
    if rating_raw is not None:
        try:
            rating = int(float(rating_raw))
        except (TypeError, ValueError):
            pass

    isbn = entry.get("BKDisplayID") or entry.get("itemId") or None

    # Try to resolve the actual file path
    file_path: str | None = None
    for key in ("BKBookFilePath", "BKPathRelativeToiCloudContainer", "path"):
        raw = entry.get(key)
        if raw:
            p = Path(str(raw)).expanduser()
            if not p.is_absolute():
                p = _ICLOUD_BOOKS / p
            if p.exists():
                file_path = str(p)
                break

    return BookMetadata(
        book_id=_book_id(title, author),
        title=title,
        author=author,
        genre=genre,
        description=description,
        rating=rating,
        isbn=str(isbn) if isbn else None,
        file_path=file_path,
    )


def load_apple_books_metadata() -> list[BookMetadata]:
    """Return metadata for all books in the Apple Books library."""
    plist_path: Path | None = None
    for candidate in _PLIST_CANDIDATES:
        if candidate.exists():
            plist_path = candidate
            break

    if plist_path is None:
        return []

    with plist_path.open("rb") as f:
        data = plistlib.load(f)

    entries: list[dict] = []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        for key in ("Books", "books", "items"):
            if key in data and isinstance(data[key], list):
                entries = data[key]
                break

    books: list[BookMetadata] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        book = _parse_entry(entry)
        if book:
            books.append(book)

    return books
