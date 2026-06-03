"""Embedding provider abstraction: OpenAI, Voyage AI, or Ollama.

Note on input_type: Voyage AI embeds documents and queries differently for
better retrieval — passages indexed as "document", search strings as "query".
OpenAI and Ollama ignore this distinction (it's a no-op for them).
"""
from __future__ import annotations

import time
from typing import Literal

import httpx

InputType = Literal["document", "query"]

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


class Embedder:
    def __init__(
        self,
        provider: Literal["openai", "voyage", "ollama"] = "openai",
        model_openai: str = "text-embedding-3-small",
        model_voyage: str = "voyage-3",
        model_ollama: str = "nomic-embed-text",
        openai_api_key: str = "",
        voyage_api_key: str = "",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.provider = provider
        self.model_openai = model_openai
        self.model_voyage = model_voyage
        self.model_ollama = model_ollama
        self._openai_key = openai_api_key
        self._voyage_key = voyage_api_key
        self._ollama_url = ollama_base_url.rstrip("/")
        self._client: "openai.OpenAI | None" = None  # type: ignore[name-defined]

    def _get_openai(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._openai_key)
        return self._client

    def embed(self, texts: list[str], input_type: InputType = "document") -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        if not texts:
            return []
        texts = [t.replace("\n", " ") for t in texts]

        if self.provider == "openai":
            return self._embed_openai(texts)
        if self.provider == "voyage":
            return self._embed_voyage(texts, input_type)
        return self._embed_ollama(texts)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        client = self._get_openai()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=self.model_openai, input=texts)
                return [item.embedding for item in resp.data]
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return []  # unreachable

    def _embed_voyage(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self._voyage_key}"}
        payload = {"input": texts, "model": self.model_voyage, "input_type": input_type}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=60) as client:
                    resp = client.post(_VOYAGE_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()["data"]
                # Voyage returns results with an `index`; sort to preserve input order.
                data.sort(key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return []  # unreachable

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=60) as client:
            for text in texts:
                resp = client.post(
                    f"{self._ollama_url}/api/embeddings",
                    json={"model": self.model_ollama, "prompt": text},
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
        return vectors

    def embed_one(self, text: str, input_type: InputType = "query") -> list[float]:
        return self.embed([text], input_type=input_type)[0]


def embedder_from_settings(settings) -> Embedder:
    return Embedder(
        provider=settings.embedding_provider,
        model_openai=settings.embedding_model_openai,
        model_voyage=settings.embedding_model_voyage,
        model_ollama=settings.embedding_model_ollama,
        openai_api_key=settings.openai_api_key,
        voyage_api_key=settings.voyage_api_key,
        ollama_base_url=settings.ollama_base_url,
    )
