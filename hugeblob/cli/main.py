"""Typer CLI for hugeblob."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown

app = typer.Typer(
    name="hugeblob",
    help="LLM-powered interface to your personal ebook library.",
    no_args_is_help=True,
)
console = Console()


def _resolve_settings(collection: Optional[str] = None):
    """Load settings, optionally overriding the active collection."""
    from hugeblob.config import get_settings

    settings = get_settings()
    if collection:
        settings.qdrant_collection = collection
    return settings


def _get_deps(collection: Optional[str] = None):
    """Lazy-load heavy dependencies so --help is instant."""
    import anthropic
    from hugeblob.embed.embedder import embedder_from_settings
    from hugeblob.store.qdrant_store import store_from_settings

    settings = _resolve_settings(collection)
    embedder = embedder_from_settings(settings)
    store = store_from_settings(settings)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return settings, embedder, store, client


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    source: str = typer.Option(
        "all", "--source", "-s",
        help="What to ingest: all | readwise | apple-books | files",
    ),
    books_dir: Optional[Path] = typer.Option(
        None, "--books-dir", "-d",
        help="Override BOOKS_DIR for the 'files' folder-scan source.",
    ),
    shelf: Optional[str] = typer.Option(
        None, "--shelf",
        help="Only ingest books on this Apple Books shelf/collection (e.g. 'Food').",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest already processed files."),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l",
        help="Max number of new files to process per run. Re-run to continue.",
    ),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c",
        help="Target collection (separate library). Defaults to QDRANT_COLLECTION.",
    ),
):
    """Ingest your library into the vector store.

    Sources:
      apple-books  read the Apple Books library directly (metadata + shelves + files)
      files        scan a folder (BOOKS_DIR) for loose EPUB/PDFs (fallback)
      readwise     sync Readwise highlights
      all          readwise + apple-books
    """
    from hugeblob.embed.embedder import embedder_from_settings
    from hugeblob.store.qdrant_store import store_from_settings
    from hugeblob.ingest.pipeline import ingest_readwise, ingest_files, ingest_apple_books

    valid = {"all", "readwise", "apple-books", "files"}
    if source not in valid:
        console.print(f"[red]Unknown source '{source}'.[/red] Choose from: {', '.join(sorted(valid))}")
        raise typer.Exit(1)

    settings = _resolve_settings(collection)
    embedder = embedder_from_settings(settings)
    store = store_from_settings(settings)

    console.print(f"[dim]Collection: {settings.qdrant_collection}[/dim]")
    log = lambda msg: console.print(f"  {msg}")

    if source in ("all", "readwise"):
        ingest_readwise(settings, store, embedder, log=log)

    if source in ("all", "apple-books"):
        ingest_apple_books(settings, store, embedder, shelf=shelf, force=force, limit=limit, log=log)

    if source == "files":
        ingest_files(
            settings, store, embedder,
            books_dir=books_dir, force=force, limit=limit, shelf=shelf, log=log,
        )

    console.print(f"\n[green]Done.[/green] Total indexed chunks: {store.count()}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
    highlights_only: bool = typer.Option(False, "--highlights", "-H", help="Only search highlights"),
    author: Optional[str] = typer.Option(None, "--author", "-a", help="Filter by author"),
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Filter by genre"),
    shelf: Optional[str] = typer.Option(None, "--shelf", help="Filter by Apple Books shelf (e.g. 'Food')"),
    min_rating: Optional[int] = typer.Option(None, "--min-rating", "-r", help="Minimum rating (1-5)"),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection to search. Defaults to QDRANT_COLLECTION.",
    ),
):
    """Semantic search across your library."""
    from hugeblob.query.search import search as do_search

    settings, embedder, store, _ = _get_deps(collection)
    results = do_search(
        query,
        store,
        embedder,
        limit=limit,
        highlights_only=highlights_only,
        author=author,
        genre=genre,
        shelf=shelf,
        min_rating=min_rating,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit()

    for i, r in enumerate(results, 1):
        source_tag = "[cyan]highlight[/cyan]" if r.source.value == "highlight" else "[blue]passage[/blue]"
        title_line = f"[bold]{r.title}[/bold] — {r.author}  {source_tag}  score={r.score:.3f}"
        body = r.text[:400] + ("…" if len(r.text) > 400 else "")
        footer_parts = []
        if r.page_number:
            footer_parts.append(f"p.{r.page_number}")
        if r.highlight_note:
            footer_parts.append(f"note: {r.highlight_note}")
        footer = "  ·  ".join(footer_parts)

        console.print(Panel(
            f"{title_line}\n\n{body}" + (f"\n\n[dim]{footer}[/dim]" if footer else ""),
            title=f"#{i}",
            border_style="dim",
        ))


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@app.command()
def chat(
    book: Optional[str] = typer.Option(None, "--book", "-b", help="Focus chat on a specific book title."),
    highlights_only: bool = typer.Option(False, "--highlights", "-H", help="Pull from highlights only."),
    shelf: Optional[str] = typer.Option(None, "--shelf", help="Restrict chat to an Apple Books shelf (e.g. 'Food')."),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection to chat with. Defaults to QDRANT_COLLECTION.",
    ),
):
    """Interactive chat session grounded in your library."""
    from hugeblob.query.search import search as do_search
    from hugeblob.query.llm import ask, ConversationHistory

    settings, embedder, store, client = _get_deps(collection)
    history = ConversationHistory()

    console.print(Panel(
        f"[bold]hugeblob chat[/bold]  [dim](collection: {settings.qdrant_collection})[/dim]\n"
        "Ask anything about your library. Type [bold]exit[/bold] or Ctrl-C to quit.\n"
        "[dim]/clear[/dim] resets conversation history.",
        border_style="blue",
    ))

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if question.strip() == "/clear":
            history.clear()
            console.print("[dim]History cleared.[/dim]")
            continue
        if not question.strip():
            continue

        results = do_search(
            question,
            store,
            embedder,
            limit=8,
            highlights_only=highlights_only,
            title=book,
            shelf=shelf,
        )

        if not results:
            console.print("[yellow]No relevant passages found in the library.[/yellow]")
            continue

        console.print(f"\n[dim]Retrieved {len(results)} passages from {len({r.book_id for r in results})} books.[/dim]")
        console.print("\n[bold green]Assistant[/bold green]")

        full_reply = []
        for chunk in ask(question, results, client, model=settings.claude_model, history=history, stream=True):
            console.print(chunk, end="")
            full_reply.append(chunk)
        console.print()

        # Update history with non-streamed version (already handled inside ask for non-stream)
        if not history.messages or history.messages[-1]["role"] != "assistant":
            history.add_assistant("".join(full_reply))


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

@app.command()
def synthesize(
    topic: str = typer.Argument(..., help="Topic to synthesize across your library."),
    limit: int = typer.Option(12, "--limit", "-n", help="Number of passages to gather."),
    shelf: Optional[str] = typer.Option(None, "--shelf", help="Restrict to an Apple Books shelf (e.g. 'Food')."),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection to synthesize from. Defaults to QDRANT_COLLECTION.",
    ),
):
    """Cross-book synthesis: find themes and connections on a topic."""
    from hugeblob.query.search import search as do_search
    from hugeblob.query.llm import synthesize as do_synthesize

    settings, embedder, store, client = _get_deps(collection)

    console.print(f"[dim]Gathering passages about: {topic}...[/dim]")
    results = do_search(topic, store, embedder, limit=limit, shelf=shelf)

    if not results:
        console.print("[yellow]No relevant passages found.[/yellow]")
        raise typer.Exit()

    book_titles = sorted({r.title for r in results})
    console.print(f"[dim]Sources: {', '.join(book_titles)}[/dim]\n")

    with console.status("Synthesizing..."):
        analysis = do_synthesize(topic, results, client, model=settings.claude_model)

    console.print(Markdown(analysis))


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@app.command()
def stats(
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection to report on. Defaults to QDRANT_COLLECTION.",
    ),
):
    """Show library statistics."""
    from hugeblob.store.qdrant_store import store_from_settings
    import json

    settings = _resolve_settings(collection)
    store = store_from_settings(settings)

    total = store.count()
    state = {}
    if settings.state_file.exists():
        state = json.loads(settings.state_file.read_text())

    table = Table(title="Library Stats", show_header=False, border_style="dim")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Collection", settings.qdrant_collection)
    table.add_row("Total indexed chunks", str(total))
    table.add_row("Ingested files", str(len(state.get("ingested_files", {}))))
    table.add_row("Last Readwise sync", state.get("readwise_last_sync") or "never")
    table.add_row("Data directory", str(settings.data_dir))
    table.add_row("Books directory", str(settings.books_dir))
    table.add_row("Embedding provider", settings.embedding_provider)
    table.add_row("Claude model", settings.claude_model)
    console.print(table)


# ---------------------------------------------------------------------------
# shelves
# ---------------------------------------------------------------------------

@app.command()
def shelves():
    """List the Apple Books shelves (Collections) found on this Mac."""
    from hugeblob.ingest.apple_books import list_shelves as get_shelves

    rows = get_shelves()
    if not rows:
        console.print(
            "[yellow]No Apple Books shelves found.[/yellow]\n"
            "[dim]Either Apple Books isn't on this machine, its library is in an "
            "unrecognised location, or no user Collections exist. Run "
            "`hugeblob doctor` to probe the library location.[/dim]"
        )
        raise typer.Exit()

    table = Table(title="Apple Books shelves", border_style="dim")
    table.add_column("Shelf", style="bold")
    table.add_column("Books", justify="right")
    for name, count in rows:
        table.add_row(name, str(count))
    console.print(table)
    console.print("\n[dim]Use with:  hugeblob ingest --source apple-books --shelf \"<name>\"[/dim]")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

@app.command()
def doctor():
    """Probe where the Apple Books library lives and what this tool can read."""
    from hugeblob.ingest import apple_books as ab

    console.print("[bold]Apple Books library probe[/bold]\n")

    db = ab._find_library_db()
    if db:
        console.print(f"[green]SQLite library found:[/green] {db}")
    else:
        console.print(
            f"[yellow]No SQLite library at[/yellow] {ab._IBOOKSX_BKLIBRARY}"
        )

    plist = next((c for c in ab._PLIST_CANDIDATES if c.exists()), None)
    if plist:
        console.print(f"[green]Legacy plist found:[/green] {plist}")

    books = ab.load_apple_books()
    with_files = [b for b in books if b.file_path]
    console.print(f"\nBooks read: [bold]{len(books)}[/bold]")
    console.print(f"With a resolvable file path: [bold]{len(with_files)}[/bold]")
    shelf_rows = ab.list_shelves()
    console.print(f"Shelves (Collections): [bold]{len(shelf_rows)}[/bold]")
    if shelf_rows:
        preview = ", ".join(f"{n} ({c})" for n, c in shelf_rows[:8])
        console.print(f"[dim]{preview}[/dim]")

    if not books:
        console.print(
            "\n[yellow]Nothing found.[/yellow] Your Apple Books data may be in a "
            "different location. Please share the output of:\n"
            "[dim]find ~/Library -iname '*.sqlite' 2>/dev/null | grep -iE 'book'[/dim]"
        )


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve():
    """Start the MCP server (for Claude Desktop integration)."""
    from hugeblob.mcp.server import serve as run_mcp
    console.print("[dim]Starting MCP server on stdio...[/dim]")
    run_mcp()
