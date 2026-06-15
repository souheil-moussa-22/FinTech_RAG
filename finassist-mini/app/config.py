"""Application configuration loaded from environment variables."""

from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    """Centralized settings used across services and API routes."""

    app_name: str = Field(default=os.getenv("APP_NAME", "finassist-mini"))
    app_version: str = Field(default=os.getenv("APP_VERSION", "0.1.0"))
    docs_directory: Path = Field(default=BASE_DIR / os.getenv("DOCS_DIRECTORY", "data/documents"))
    chroma_directory: Path = Field(default=BASE_DIR / os.getenv("CHROMA_DIRECTORY", "data/chroma"))
    chroma_collection: str = Field(default=os.getenv("CHROMA_COLLECTION", "finassist_chunks"))
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    ollama_url: str = Field(default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    llm_model: str = Field(default=os.getenv("LLM_MODEL", "mistral"))
    retrieval_k: int = Field(default=int(os.getenv("RETRIEVAL_K", "4")), ge=1)
    chunk_size_words: int = Field(default=int(os.getenv("CHUNK_SIZE_WORDS", "500")), ge=50)
    chunk_overlap_words: int = Field(default=int(os.getenv("CHUNK_OVERLAP_WORDS", "50")), ge=0)
    llm_timeout_seconds: int = Field(default=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")), ge=1)


settings = Settings()
