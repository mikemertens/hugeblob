"""Embedding provider abstraction: OpenAI or Ollama."""
from __future__ import annotations

import time
from typing import Literal

import httpx


class Embedder:
    def __init__(
        self,
        provider: Literal["openai", "ollama"] = "openai",
        model_openai: str = "text-embedding-3-small",
        model_ollama: str = "nomic-embed-text",
        openai_api_key: str = "",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.provider = provider
        self.model_openai = model_openai
        self.model_ollama = model_ollama
        self._openai_key = openai_api_key
        self._ollama_url = ollama_base_url.rstrip("/")
        self._client: "openai.OpenAI | None" = None  # type: ignore[name-defined]

    def _get_openai(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._openai_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        if not texts:
            return []
        texts = [t.replace("\n", " ") for t in texts]

        if self.provider == "openai":
            return self._embed_openai(texts)
        return self._embed_ollama(texts)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        client = self._get_openai()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=self.model_openai, input=texts)
                return [item.embedding for item in resp.data]
            except Exception as e:
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

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def embedder_from_settings(settings) -> Embedder:
    return Embedder(
        provider=settings.embedding_provider,
        model_openai=settings.embedding_model_openai,
        model_ollama=settings.embedding_model_ollama,
        openai_api_key=settings.openai_api_key,
        ollama_base_url=settings.ollama_base_url,
    )
