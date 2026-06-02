from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocSource(str, Enum):
    FULLTEXT = "fulltext"
    HIGHLIGHT = "highlight"


class BookMetadata(BaseModel):
    book_id: str
    title: str
    author: str = ""
    genre: str = ""
    description: str = ""
    rating: Optional[int] = None
    tags: list[str] = Field(default_factory=list)
    file_path: Optional[str] = None
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    year: Optional[int] = None


class TextChunk(BaseModel):
    book_id: str
    title: str
    author: str = ""
    genre: str = ""
    rating: Optional[int] = None
    source: DocSource
    text: str
    chunk_index: int = 0
    page_number: Optional[int] = None
    file_path: Optional[str] = None
    # highlight-specific
    highlight_id: Optional[str] = None
    highlight_note: Optional[str] = None
    highlight_tags: list[str] = Field(default_factory=list)
    readwise_url: Optional[str] = None


class SearchResult(BaseModel):
    score: float
    text: str
    title: str
    author: str
    source: DocSource
    book_id: str
    page_number: Optional[int] = None
    file_path: Optional[str] = None
    highlight_note: Optional[str] = None
    readwise_url: Optional[str] = None

    def format(self) -> str:
        lines = [f"[{self.source.value}] {self.title} — {self.author}  (score: {self.score:.3f})"]
        lines.append(self.text.strip())
        if self.highlight_note:
            lines.append(f"Note: {self.highlight_note}")
        if self.page_number:
            lines.append(f"Page {self.page_number}")
        return "\n".join(lines)
