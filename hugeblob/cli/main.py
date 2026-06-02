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


def _get_deps():
    """Lazy-load heavy dependencies so --help is instant."""
    import anthropic
    from hugeblob.config import get_settings
    from hugeblob.embed.embedder import embedder_from_settings
    from hugeblob.store.qdrant_store import store_from_settings

    settings = get_settings()
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
        help="What to ingest: all | readwise | files",
    ),
    books_dir: Optional[Path] = typer.Option(
        None, "--books-dir", "-d",
        help="Override the BOOKS_DIR setting for this run.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest already processed files."),
):
    """Ingest your library into the vector store."""
    from hugeblob.config import get_settings
    from hugeblob.embed.embedder import embedder_from_settings
    from hugeblob.store.qdrant_store import store_from_settings
    from hugeblob.ingest.pipeline import ingest_readwise, ingest_files

    settings = get_settings()
    embedder = embedder_from_settings(settings)
    store = store_from_settings(settings)

    log = lambda msg: console.print(f"  {msg}")

    if source in ("all", "readwise"):
        ingest_readwise(settings, store, embedder, log=log)

    if source in ("all", "files"):
        ingest_files(settings, store, embedder, books_dir=books_dir, force=force, log=log)

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
    min_rating: Optional[int] = typer.Option(None, "--min-rating", "-r", help="Minimum rating (1-5)"),
):
    """Semantic search across your library."""
    from hugeblob.query.search import search as do_search

    settings, embedder, store, _ = _get_deps()
    results = do_search(
        query,
        store,
        embedder,
        limit=limit,
        highlights_only=highlights_only,
        author=author,
        genre=genre,
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
):
    """Interactive chat session grounded in your library."""
    from hugeblob.query.search import search as do_search
    from hugeblob.query.llm import ask, ConversationHistory

    settings, embedder, store, client = _get_deps()
    history = ConversationHistory()

    console.print(Panel(
        "[bold]hugeblob chat[/bold]\n"
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
):
    """Cross-book synthesis: find themes and connections on a topic."""
    from hugeblob.query.search import search as do_search
    from hugeblob.query.llm import synthesize as do_synthesize

    settings, embedder, store, client = _get_deps()

    console.print(f"[dim]Gathering passages about: {topic}...[/dim]")
    results = do_search(topic, store, embedder, limit=limit)

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
def stats():
    """Show library statistics."""
    from hugeblob.config import get_settings
    from hugeblob.store.qdrant_store import store_from_settings
    import json

    settings = get_settings()
    store = store_from_settings(settings)

    total = store.count()
    state = {}
    if settings.state_file.exists():
        state = json.loads(settings.state_file.read_text())

    table = Table(title="Library Stats", show_header=False, border_style="dim")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total indexed chunks", str(total))
    table.add_row("Ingested files", str(len(state.get("ingested_files", {}))))
    table.add_row("Last Readwise sync", state.get("readwise_last_sync") or "never")
    table.add_row("Data directory", str(settings.data_dir))
    table.add_row("Books directory", str(settings.books_dir))
    table.add_row("Embedding provider", settings.embedding_provider)
    table.add_row("Claude model", settings.claude_model)
    console.print(table)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve():
    """Start the MCP server (for Claude Desktop integration)."""
    from hugeblob.mcp.server import serve as run_mcp
    console.print("[dim]Starting MCP server on stdio...[/dim]")
    run_mcp()
