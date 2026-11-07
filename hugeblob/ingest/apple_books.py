"""Read the macOS Apple Books library (SQLite), with a legacy plist fallback.

Modern Apple Books (iBooksX, 2018+) stores its catalogue in a SQLite database
under ~/Library/Containers/com.apple.iBooksX/Data/Documents/BKLibrary/. This
reader pulls title/author/genre/path plus each book's Collections ("shelves"),
opening the DB read-only + immutable so it works even while Books.app is open.

If no SQLite library is found (very old macOS), it falls back to the legacy
Books.plist reader.
"""
from __future__ import annotations

import hashlib
import plistlib
import sqlite3
from pathlib import Path
from typing import Any

from hugeblob.models import BookMetadata

_IBOOKSX_BKLIBRARY = (
    Path.home() / "Library/Containers/com.apple.iBooksX/Data/Documents/BKLibrary"
)

# The library SQLite DB has moved across macOS versions — search several roots.
_SQLITE_DIR_CANDIDATES = [
    _IBOOKSX_BKLIBRARY,
    Path.home() / "Library/Containers/com.apple.iBooksX/Data/Documents",
    Path.home() / "Library/Group Containers/group.com.apple.iBooks/Documents/BKLibrary",
    Path.home() / "Library/Containers/com.apple.BKAgentService/Data/Documents/iBooks/BKLibrary",
]

# Legacy plist locations (pre-2018 iBooks); used only if no SQLite DB is found.
_PLIST_CANDIDATES = [
    Path.home() / "Library/Containers/com.apple.BKAgentService/Data/Documents/iBooks/Books/Books.plist",
    Path.home() / "Library/Group Containers/group.com.apple.iBooks/Documents/iBooks/Books/Books.plist",
    Path.home() / "Library/Containers/com.apple.iBooksX/Data/Documents/iBooks/Books/Books.plist",
]
_ICLOUD_BOOKS = Path.home() / "Library/Mobile Documents/iCloud~com~apple~iBooks/Documents"


def _book_id_from_title(title: str, author: str) -> str:
    return hashlib.md5(f"{title}|{author}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# SQLite reader (modern Apple Books)
# --------------------------------------------------------------------------- #

def _find_library_db() -> Path | None:
    for base in _SQLITE_DIR_CANDIDATES:
        if base.exists():
            dbs = sorted(base.glob("BKLibrary*.sqlite"))
            if dbs:
                return dbs[0]
    return None


def _connect_ro(path: Path) -> sqlite3.Connection:
    # immutable=1 lets us read even while Books.app holds a WAL lock.
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _load_shelves(con: sqlite3.Connection) -> dict[str, list[str]]:
    """Map asset_id -> list of user Collection (shelf) titles.

    Written defensively: the exact collection/membership schema has shifted
    across macOS versions, so we introspect columns and skip gracefully if the
    linkage can't be determined (books still ingest, just without shelf tags).
    """
    coll_cols = _columns(con, "ZBKCOLLECTION")
    mem_cols = _columns(con, "ZBKCOLLECTIONMEMBER")
    if "ZTITLE" not in coll_cols or "ZASSETID" not in mem_cols:
        return {}

    # Membership -> collection foreign key
    if "ZCOLLECTION" in mem_cols and "Z_PK" in coll_cols:
        join = "c.Z_PK = m.ZCOLLECTION"
    elif "ZCOLLECTIONID" in mem_cols and "ZCOLLECTIONID" in coll_cols:
        join = "c.ZCOLLECTIONID = m.ZCOLLECTIONID"
    else:
        return {}

    # Hide built-in/deleted collections where those flags exist.
    where = []
    if "ZISDELETED" in coll_cols:
        where.append("COALESCE(c.ZISDELETED, 0) = 0")
    if "ZISHIDDEN" in coll_cols:
        where.append("COALESCE(c.ZISHIDDEN, 0) = 0")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    sql = (
        f"SELECT c.ZTITLE AS title, m.ZASSETID AS asset_id "
        f"FROM ZBKCOLLECTIONMEMBER m JOIN ZBKCOLLECTION c ON {join}{where_sql}"
    )

    shelves: dict[str, list[str]] = {}
    try:
        for row in con.execute(sql):
            title = (row["title"] or "").strip()
            asset_id = (row["asset_id"] or "").strip()
            if not title or not asset_id:
                continue
            bucket = shelves.setdefault(asset_id, [])
            if title not in bucket:
                bucket.append(title)
    except sqlite3.Error:
        return {}
    return shelves


def _load_from_sqlite() -> list[BookMetadata]:
    db = _find_library_db()
    if db is None:
        return []

    con = _connect_ro(db)
    try:
        cols = _columns(con, "ZBKLIBRARYASSET")
        if "ZTITLE" not in cols:
            return []

        shelves_by_asset = _load_shelves(con)

        field_map = {
            "asset_id": "ZASSETID",
            "title": "ZTITLE",
            "author": "ZAUTHOR",
            "genre": "ZGENRE",
            "path": "ZPATH",
            "description": "ZBOOKDESCRIPTION",
            "rating": "ZRATING",
        }
        present = {alias: col for alias, col in field_map.items() if col in cols}
        select_sql = ", ".join(f"{col} AS {alias}" for alias, col in present.items())
        rows = con.execute(f"SELECT {select_sql} FROM ZBKLIBRARYASSET").fetchall()
    finally:
        con.close()

    books: list[BookMetadata] = []
    for row in rows:
        d = dict(row)
        title = (d.get("title") or "").strip()
        if not title:
            continue

        asset_id = (d.get("asset_id") or "").strip()
        path = (d.get("path") or "").strip() or None

        rating = None
        if d.get("rating") is not None:
            try:
                rating = int(float(d["rating"]))
            except (TypeError, ValueError):
                rating = None

        book_id = f"ab_{asset_id}" if asset_id else _book_id_from_title(title, d.get("author") or "")

        books.append(
            BookMetadata(
                book_id=book_id,
                title=title,
                author=(d.get("author") or "").strip(),
                genre=(d.get("genre") or "").strip(),
                description=(d.get("description") or "").strip(),
                rating=rating,
                file_path=path,
                shelves=shelves_by_asset.get(asset_id, []),
            )
        )
    return books


# --------------------------------------------------------------------------- #
# Legacy plist reader (fallback)
# --------------------------------------------------------------------------- #

def _parse_plist_entry(entry: dict[str, Any]) -> BookMetadata | None:
    title = (entry.get("BKBookTitle") or entry.get("title") or entry.get("itemName") or "").strip()
    if not title:
        return None

    author = (entry.get("BKBookAuthor") or entry.get("artistName") or entry.get("author") or "").strip()
    genre = (entry.get("BKBookGenre") or entry.get("primaryGenreName") or entry.get("genre") or "").strip()
    description = (entry.get("BKBookDescription") or entry.get("description") or "").strip()

    rating_raw = entry.get("BKBookRating") or entry.get("rating")
    rating: int | None = None
    if rating_raw is not None:
        try:
            rating = int(float(rating_raw))
        except (TypeError, ValueError):
            pass

    isbn = entry.get("BKDisplayID") or entry.get("itemId") or None

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
        book_id=_book_id_from_title(title, author),
        title=title,
        author=author,
        genre=genre,
        description=description,
        rating=rating,
        isbn=str(isbn) if isbn else None,
        file_path=file_path,
    )


def _load_from_plist() -> list[BookMetadata]:
    plist_path = next((c for c in _PLIST_CANDIDATES if c.exists()), None)
    if plist_path is None:
        return []

    with plist_path.open("rb") as f:
        data = plistlib.load(f)

    entries: list[dict] = []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        for key in ("Books", "books", "items"):
            if isinstance(data.get(key), list):
                entries = data[key]
                break

    books: list[BookMetadata] = []
    for entry in entries:
        if isinstance(entry, dict):
            book = _parse_plist_entry(entry)
            if book:
                books.append(book)
    return books


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_apple_books() -> list[BookMetadata]:
    """All books in the Apple Books library — SQLite preferred, plist fallback."""
    books = _load_from_sqlite()
    return books if books else _load_from_plist()


# Backwards-compatible alias (used by the folder-scan fallback in pipeline.py).
def load_apple_books_metadata() -> list[BookMetadata]:
    return load_apple_books()


def list_shelves() -> list[tuple[str, int]]:
    """Return (shelf_name, book_count) for user Collections, most-populated first."""
    counts: dict[str, int] = {}
    for b in load_apple_books():
        for s in b.shelves:
            counts[s] = counts.get(s, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
