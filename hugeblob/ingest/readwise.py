"""Fetch highlights from the Readwise v2 API."""
from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any

import httpx

from hugeblob.models import TextChunk, DocSource

_BASE = "https://readwise.io/api/v2"


def _book_id(readwise_book_id: int) -> str:
    return f"rw_{readwise_book_id}"


def _highlight_id(hid: int) -> str:
    return f"rw_h_{hid}"


def _paginate(client: httpx.Client, url: str, params: dict | None = None) -> list[dict]:
    results: list[dict] = []
    next_url: str | None = url
    p = dict(params or {})
    p.setdefault("page_size", 1000)

    while next_url:
        resp = client.get(next_url, params=p if next_url == url else None)
        resp.raise_for_status()
        body = resp.json()
        results.extend(body.get("results", []))
        next_url = body.get("next")
        if next_url:
            time.sleep(0.1)  # gentle rate limiting

    return results


def fetch_highlights(
    api_key: str,
    updated_after: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[TextChunk]]:
    """Return (raw_books, highlight_chunks) from Readwise."""
    headers = {"Authorization": f"Token {api_key}"}

    with httpx.Client(headers=headers, timeout=30) as client:
        book_params: dict[str, Any] = {}
        if updated_after:
            book_params["updated__gt"] = updated_after.isoformat()
        raw_books = _paginate(client, f"{_BASE}/books/", book_params)

        book_index = {b["id"]: b for b in raw_books}

        highlight_params: dict[str, Any] = {}
        if updated_after:
            highlight_params["updated__gt"] = updated_after.isoformat()
        raw_highlights = _paginate(client, f"{_BASE}/highlights/", highlight_params)

    chunks: list[TextChunk] = []
    for h in raw_highlights:
        text = (h.get("text") or "").strip()
        if not text:
            continue

        book_rw_id = h.get("book_id")
        book = book_index.get(book_rw_id, {})

        title = (book.get("title") or "Unknown").strip()
        author = (book.get("author") or "").strip()
        genre = (book.get("category") or "").strip()
        tags = [t["name"] for t in (h.get("tags") or []) if t.get("name")]

        chunks.append(
            TextChunk(
                book_id=_book_id(book_rw_id) if book_rw_id else hashlib.md5(title.encode()).hexdigest()[:16],
                title=title,
                author=author,
                genre=genre,
                source=DocSource.HIGHLIGHT,
                text=text,
                highlight_id=_highlight_id(h["id"]),
                highlight_note=(h.get("note") or "").strip() or None,
                highlight_tags=tags,
                readwise_url=h.get("url") or None,
            )
        )

    return raw_books, chunks
