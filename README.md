# hugeblob

LLM-powered interface to your personal ebook library. Ingest 2000+ EPUBs/PDFs + Readwise highlights into a local vector store, then chat, search, and synthesize across everything.

## What it does

- **Ingests** EPUB/PDF files from your local library, Readwise highlights, and Apple Books metadata
- **Searches** semantically across full text and highlights with author/genre/rating filters
- **Chats** with your library — Q&A grounded in actual passages with citations
- **Synthesizes** cross-book analysis: themes, comparisons, connections across authors
- **Exposes** everything as an MCP server for use inside Claude Desktop / Claude Code

## Quick start

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Install

```bash
pip install uv
uv pip install -e .
```

Or with `uv` (recommended):

```bash
uv sync
```

### 3. Configure

```bash
cp .env.example .env
# edit .env with your API keys and books directory
```

Minimum required keys:
- `ANTHROPIC_API_KEY` — for Claude Q&A
- `OPENAI_API_KEY` — for embeddings (default; or use Voyage / Ollama — see [Embedding providers](#embedding-providers))
- `READWISE_API_KEY` — for highlights (optional but recommended)
- `BOOKS_DIR` — path to your EPUB/PDF folder

### 4. Ingest

```bash
# Check that your Apple Books library is found first
hugeblob doctor

# Fast start: highlights only
hugeblob ingest --source readwise

# Read the Apple Books library directly (metadata + shelves + files)
hugeblob ingest --source apple-books --limit 500   # re-run until done

# Everything (readwise + apple-books)
hugeblob ingest
```

Apple Books is the primary source: hugeblob reads the app's own SQLite library
to get each book's title/author/genre, its Collection ("shelf") names, and the
real path to the EPUB/PDF file — no need to point at a folder. (Loose files
outside Apple Books can still be indexed with `--source files --books-dir ...`.)

### 5. Use it

```bash
# Discover your Apple Books shelves
hugeblob shelves

# Search (optionally scoped to a shelf)
hugeblob search "the role of narrative in understanding causality"
hugeblob search "what can I braise?" --shelf "Food"
hugeblob search "attention" --author "Kahneman" --highlights

# Chat
hugeblob chat
hugeblob chat --shelf "Food"          # only your cookbooks
hugeblob chat --book "Thinking, Fast and Slow"

# Cross-book synthesis
hugeblob synthesize "how do different authors approach emergence?"
hugeblob synthesize "techniques for building flavor" --shelf "Food"

# Library stats
hugeblob stats
```

## Shelves vs. collections

Two different ways to slice your library — don't confuse them:

- **Shelf** (`--shelf`) = an Apple Books *Collection* (the groupings in the
  Books sidebar, like "Food"). Every book is tagged with its shelves at ingest,
  so you can filter any query by shelf without a separate index.
- **Collection** (`--collection` / `-c`) = a separate hugeblob/Qdrant *index*.
  Fully isolated storage; queries never cross over; each tracks its own
  ingestion state.

For most cases, shelves are all you need — index everything once, filter by
shelf at query time:

```bash
hugeblob ingest --source all                 # tags every book with its shelves
hugeblob chat --shelf "Food"                 # only cookbooks
hugeblob synthesize "flavor pairing" --shelf "Food"
```

Use a separate **collection** when you want a shelf stored as its own isolated
index (smaller, faster, never mixed):

```bash
# Ingest just the Food shelf into a dedicated "cookbooks" index
hugeblob ingest --source apple-books --shelf "Food" -c cookbooks

# Everything else into the default "library" index
hugeblob ingest --source all

# Query one or the other
hugeblob chat -c cookbooks
hugeblob chat                      # main library (default collection)
hugeblob stats -c cookbooks
```

The folder fallback still works for loose files outside Apple Books:

```bash
hugeblob ingest -c extras --source files --books-dir ~/LooseBooks --limit 500
```


Notes:
- Collections are created on first ingest — no setup step needed.
- A collection's docs all share one embedding dimension, so don't switch
  embedding providers mid-collection (see [Embedding providers](#embedding-providers)).
- Skipping `--source readwise` (as above) means no highlights are pulled —
  ideal for cookbooks.

## MCP server (Claude Desktop)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hugeblob": {
      "command": "hugeblob-mcp",
      "args": [],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENAI_API_KEY": "sk-...",
        "READWISE_API_KEY": "...",
        "BOOKS_DIR": "/Users/you/Books"
      }
    }
  }
}
```

Then in Claude Desktop you can say:
- *"Search my library for passages about emergence in complex systems"*
- *"What do my highlights say about habit formation?"*
- *"Synthesize how the books in my library approach the question of consciousness"*
- *"List all my 5-star books in the philosophy genre"*

### Multiple collections in Claude Desktop

To query separate libraries from Claude Desktop, register one MCP server per
collection — each just sets a different `QDRANT_COLLECTION`:

```json
{
  "mcpServers": {
    "hugeblob": {
      "command": "hugeblob-mcp",
      "env": { "QDRANT_COLLECTION": "library", "OPENAI_API_KEY": "sk-...", "ANTHROPIC_API_KEY": "sk-ant-..." }
    },
    "hugeblob-cookbooks": {
      "command": "hugeblob-mcp",
      "env": { "QDRANT_COLLECTION": "cookbooks", "OPENAI_API_KEY": "sk-...", "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

Claude will see both tool sets and you can ask it to search whichever one you mean.

### MCP tools

| Tool | Description |
|------|-------------|
| `search_library` | Semantic search across full text + highlights |
| `search_highlights` | Search only your Readwise annotations |
| `synthesize_topic` | Cross-book thematic analysis |
| `list_books` | Browse catalogue with filters |
| `get_book_context` | Targeted search within a single book |

## Architecture

```
Apple Books metadata  ──┐
Readwise API           ──┤──► ingestion pipeline ──► Qdrant (local)
EPUB / PDF files       ──┘    (embed w/ OpenAI          │
                               or Ollama)                │
                                                         ▼
                                              Claude API (w/ prompt caching)
                                                         │
                                              ┌──────────┴──────────┐
                                              CLI (typer + rich)   MCP server
```

## Embedding providers

Claude has no embeddings API, so the vectorization step uses a separate
provider. Three are supported — pick one with `EMBEDDING_PROVIDER`:

| Provider | Model | Dim | Cost / 1M tok | Notes |
|----------|-------|-----|---------------|-------|
| `openai` (default) | text-embedding-3-small | 1536 | $0.02 | Reliable, widely used |
| `voyage` | voyage-3 | 1024 | $0.06 | Anthropic-invested; stronger retrieval quality; uses document/query input types |
| `ollama` | nomic-embed-text | 768 | free | Fully local, nothing leaves your machine |

**Switching providers requires three coordinated changes** — `EMBEDDING_PROVIDER`,
the matching `EMBEDDING_DIM` (see table), and a fresh Qdrant collection
(set a new `QDRANT_COLLECTION` or wipe the volume), since you cannot mix
vector dimensions in one collection. After switching, re-ingest.

### Voyage AI

```bash
# .env
EMBEDDING_PROVIDER=voyage
VOYAGE_API_KEY=pa-...
EMBEDDING_DIM=1024
```

Voyage embeds passages and search queries differently (`input_type` of
`document` vs `query`) for better retrieval — hugeblob sets this automatically.

### Ollama (local)

```bash
ollama pull nomic-embed-text
```

```bash
# .env
EMBEDDING_PROVIDER=ollama
EMBEDDING_DIM=768
```

## Embedding costs (one-time)

At OpenAI `text-embedding-3-small` ($0.02/1M tokens):

| Scale | OpenAI | Voyage voyage-3 |
|-------|--------|-----------------|
| Highlights only (~50k highlights) | < $0.01 | ~$0.03 |
| 500 books | ~$1 | ~$3 |
| 2,000 books | ~$4 | ~$12 |

This is a one-time cost. Incremental re-ingestion only processes changed files.
Ollama is free at any scale (compute is local).
