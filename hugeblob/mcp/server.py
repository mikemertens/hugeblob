"""MCP server — exposes library search and synthesis as Claude Desktop tools.

Add to ~/Library/Application Support/Claude/claude_desktop_config.json:

    {
      "mcpServers": {
        "hugeblob": {
          "command": "hugeblob-mcp",
          "args": [],
          "env": { "ANTHROPIC_API_KEY": "...", "OPENAI_API_KEY": "..." }
        }
      }
    }
"""
from __future__ import annotations

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hugeblob", instructions=(
    "Search and synthesize across a personal ebook library of 2000+ books. "
    "Use search_library for semantic passage lookup, search_highlights for curated "
    "Readwise highlights, synthesize_topic for cross-book analysis, and list_books "
    "to browse the catalogue."
))


def _deps():
    from hugeblob.config import get_settings
    from hugeblob.embed.embedder import embedder_from_settings
    from hugeblob.store.qdrant_store import store_from_settings
    settings = get_settings()
    return settings, embedder_from_settings(settings), store_from_settings(settings)


@mcp.tool()
def search_library(
    query: str,
    limit: int = 8,
    author: Optional[str] = None,
    genre: Optional[str] = None,
    min_rating: Optional[int] = None,
) -> str:
    """Semantic search across full book text and highlights in the library.

    Args:
        query: Natural language search query.
        limit: Max results to return (default 8).
        author: Filter to a specific author name.
        genre: Filter to a genre (e.g. "fiction", "science").
        min_rating: Minimum Apple Books star rating (1-5).
    """
    from hugeblob.query.search import search

    settings, embedder, store = _deps()
    results = search(
        query, store, embedder,
        limit=limit, author=author, genre=genre, min_rating=min_rating,
    )

    if not results:
        return "No relevant passages found."

    parts = []
    for i, r in enumerate(results, 1):
        source = "highlight" if r.source.value == "highlight" else "passage"
        note = f"\nNote: {r.highlight_note}" if r.highlight_note else ""
        page = f"\n(page {r.page_number})" if r.page_number else ""
        parts.append(
            f"[{i}] ({source}) \"{r.title}\" by {r.author}  score={r.score:.3f}\n"
            f"{r.text.strip()}{note}{page}"
        )

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def search_highlights(query: str, limit: int = 8) -> str:
    """Search only the Readwise highlights (your personal curated annotations).

    Args:
        query: Natural language search query.
        limit: Max results to return.
    """
    from hugeblob.query.search import search
    from hugeblob.models import DocSource

    settings, embedder, store = _deps()
    results = search(query, store, embedder, limit=limit, highlights_only=True)

    if not results:
        return "No matching highlights found."

    parts = []
    for i, r in enumerate(results, 1):
        note = f"\nYour note: {r.highlight_note}" if r.highlight_note else ""
        url = f"\n{r.readwise_url}" if r.readwise_url else ""
        parts.append(
            f"[{i}] \"{r.title}\" by {r.author}\n"
            f"{r.text.strip()}{note}{url}"
        )

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def synthesize_topic(topic: str, limit: int = 12) -> str:
    """Cross-book synthesis: find patterns, themes, and connections on a topic.

    Gathers relevant passages from multiple books then uses Claude to synthesize
    a thematic analysis comparing how different authors address the topic.

    Args:
        topic: Topic or question to synthesize across the library.
        limit: Number of passages to gather before synthesis.
    """
    import anthropic
    from hugeblob.query.search import search
    from hugeblob.query.llm import synthesize

    settings, embedder, store = _deps()
    results = search(topic, store, embedder, limit=limit)

    if not results:
        return "No relevant passages found for this topic."

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return synthesize(topic, results, client, model=settings.claude_model)


@mcp.tool()
def list_books(
    author: Optional[str] = None,
    genre: Optional[str] = None,
    min_rating: Optional[int] = None,
    limit: int = 50,
) -> str:
    """Browse the book catalogue with optional filters.

    Args:
        author: Filter by author name (substring match).
        genre: Filter by genre.
        min_rating: Minimum star rating.
        limit: Max books to return.
    """
    settings, _, store = _deps()
    books = store.distinct_books()

    filtered = []
    for b in books:
        if author and author.lower() not in (b.get("author") or "").lower():
            continue
        if genre and genre.lower() not in (b.get("genre") or "").lower():
            continue
        if min_rating is not None:
            r = b.get("rating")
            if r is None or int(r) < min_rating:
                continue
        filtered.append(b)
        if len(filtered) >= limit:
            break

    if not filtered:
        return "No books match those filters."

    lines = []
    for b in sorted(filtered, key=lambda x: x.get("title", "")):
        rating = f"  ★{b['rating']}" if b.get("rating") else ""
        genre_str = f"  [{b['genre']}]" if b.get("genre") else ""
        lines.append(f"• {b.get('title', '?')} — {b.get('author', '?')}{rating}{genre_str}")

    return f"{len(filtered)} books:\n\n" + "\n".join(lines)


@mcp.tool()
def get_book_context(title: str, query: str, limit: int = 6) -> str:
    """Get specific passages from a single book matching a query.

    Args:
        title: Exact or partial book title.
        query: What to look for within this book.
        limit: Number of passages to return.
    """
    from hugeblob.query.search import search

    settings, embedder, store = _deps()
    results = search(query, store, embedder, limit=limit, title=title)

    if not results:
        results = search(query, store, embedder, limit=limit)
        results = [r for r in results if title.lower() in r.title.lower()]

    if not results:
        return f"No passages found in \"{title}\" for that query."

    parts = []
    for i, r in enumerate(results, 1):
        page = f"  (page {r.page_number})" if r.page_number else ""
        parts.append(f"[{i}]{page}\n{r.text.strip()}")

    header = f"\"{results[0].title}\" by {results[0].author}\n\n"
    return header + "\n\n---\n\n".join(parts)


def serve() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    serve()
