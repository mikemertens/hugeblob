from __future__ import annotations

from pathlib import Path
from typing import Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    readwise_api_key: str = ""

    # Paths
    books_dir: Path = Path.home() / "Books"
    data_dir: Path = Path.home() / ".hugeblob"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "library"

    # Embeddings
    embedding_provider: Literal["openai", "ollama"] = "openai"
    embedding_model_openai: str = "text-embedding-3-small"
    embedding_model_ollama: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    embedding_dim: int = 1536  # text-embedding-3-small; set 768 for nomic-embed-text

    # Claude
    claude_model: str = "claude-sonnet-4-6"

    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 50

    @field_validator("books_dir", "data_dir", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser()

    def ensure_data_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_file(self) -> Path:
        return self.data_dir / "state.json"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
