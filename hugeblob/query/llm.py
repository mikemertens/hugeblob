"""Claude-powered Q&A and synthesis over retrieved passages.

Uses prompt caching on the system prompt and retrieved context so that
repeat queries against the same result set are fast and cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import anthropic

from hugeblob.models import SearchResult

_SYSTEM = (
    "You are a knowledgeable assistant with access to a personal book library. "
    "Answer questions grounded in the provided passages. "
    "When citing a source, include the book title and author. "
    "If the passages do not contain enough information, say so clearly. "
    "Be concise unless the user asks for depth."
)


@dataclass
class ConversationHistory:
    messages: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def clear(self) -> None:
        self.messages.clear()


def _format_context(results: list[SearchResult]) -> str:
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        source_label = "Highlight" if r.source.value == "highlight" else "Passage"
        header = f"[{i}] {source_label} from \"{r.title}\" by {r.author}"
        body = r.text.strip()
        note = f"\nNote: {r.highlight_note}" if r.highlight_note else ""
        page = f"\n(page {r.page_number})" if r.page_number else ""
        parts.append(f"{header}\n{body}{note}{page}")
    return "\n\n---\n\n".join(parts)


def ask(
    question: str,
    results: list[SearchResult],
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    history: ConversationHistory | None = None,
    stream: bool = False,
) -> str | Iterator[str]:
    """Ask Claude a question grounded in the retrieved passages."""
    context_text = _format_context(results)

    user_content = f"Relevant passages from my library:\n\n{context_text}\n\n---\n\nQuestion: {question}"

    messages: list[dict] = []
    if history:
        messages.extend(history.messages)
    messages.append({"role": "user", "content": user_content})

    # Cache the system prompt (always reused) and context block (reused within a session)
    system = [
        {
            "type": "text",
            "text": _SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    if stream:
        return _stream(client, model, system, messages)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    reply = response.content[0].text

    if history:
        history.add_user(user_content)
        history.add_assistant(reply)

    return reply


def _stream(
    client: anthropic.Anthropic,
    model: str,
    system: list[dict],
    messages: list[dict],
) -> Iterator[str]:
    with client.messages.stream(
        model=model,
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        yield from stream.text_stream


def synthesize(
    topic: str,
    results: list[SearchResult],
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Cross-book synthesis: themes, comparisons, patterns across passages."""
    context_text = _format_context(results)
    prompt = (
        f"Using the following passages from multiple books in my library, synthesize a "
        f"thoughtful analysis of the topic: \"{topic}\"\n\n"
        f"Look for: common themes, contrasting viewpoints, interesting connections between "
        f"authors, and any evolution of ideas across sources.\n\n"
        f"Passages:\n\n{context_text}"
    )

    system = [{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}]
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
