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
- `OPENAI_API_KEY` — for embeddings (or set `EMBEDDING_PROVIDER=ollama`)
- `READWISE_API_KEY` — for highlights (optional but recommended)
- `BOOKS_DIR` — path to your EPUB/PDF folder

### 4. Ingest

```bash
# Fast start: highlights + metadata only
hugeblob ingest --source readwise

# Full library (runs in background, takes a while for 2k books)
hugeblob ingest --source files

# Everything
hugeblob ingest
```

### 5. Use it

```bash
# Search
hugeblob search "the role of narrative in understanding causality"
hugeblob search "attention mechanisms" --author "Kahneman" --highlights

# Chat
hugeblob chat
hugeblob chat --book "Thinking, Fast and Slow"

# Cross-book synthesis
hugeblob synthesize "how do different authors approach the concept of emergence?"

# Library stats
hugeblob stats
```

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

## Embedding costs

At `text-embedding-3-small` ($0.02/1M tokens):

| Scale | Estimated cost |
|-------|----------------|
| Highlights only (~50k highlights) | < $0.01 |
| 500 books | ~$1 |
| 2,000 books | ~$4 |

This is a one-time cost. Incremental re-ingestion only processes changed files.

## Local embeddings

Set `EMBEDDING_PROVIDER=ollama` and run:

```bash
ollama pull nomic-embed-text
```

Set `EMBEDDING_DIM=768` in `.env` to match nomic-embed-text's output dimension.
Note: you cannot mix embedding providers in the same Qdrant collection.
