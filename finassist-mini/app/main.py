"""FastAPI entrypoint wiring routes and service lifecycle."""

import asyncio
from contextlib import asynccontextmanager
from threading import BoundedSemaphore

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.config import settings
from app.services.chunker import ChunkerService
from app.services.embedding import EmbeddingService
from app.services.ingestion_jobs import IngestionJobService
from app.services.llm import LLMService
from app.services.observability import start_memory_tracing
from app.services.parser import PDFParserService
from app.services.rag import RAGService
from app.services.vector_store import VectorStoreService


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize shared services at startup and keep them on app state."""
    start_memory_tracing()
    settings.docs_directory.mkdir(parents=True, exist_ok=True)
    settings.chroma_directory.mkdir(parents=True, exist_ok=True)

    application.state.settings = settings
    application.state.parser_service = PDFParserService()
    application.state.chunker_service = ChunkerService(
        chunk_size_words=settings.chunk_size_words,
        chunk_overlap_words=settings.chunk_overlap_words,
    )
    application.state.embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    application.state.vector_store_service = VectorStoreService(
        persist_directory=settings.chroma_directory,
        collection_name=settings.chroma_collection,
    )
    application.state.llm_service = LLMService(
        base_url=settings.ollama_url,
        model_name=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    application.state.rag_service = RAGService(
        embedding_service=application.state.embedding_service,
        vector_store_service=application.state.vector_store_service,
        llm_service=application.state.llm_service,
        top_k=settings.retrieval_k,
        max_prompt_chars=settings.max_prompt_chars,
    )
    application.state.ingestion_job_service = IngestionJobService()
    application.state.upload_semaphore = asyncio.Semaphore(settings.upload_max_concurrency)
    application.state.chat_semaphore = asyncio.Semaphore(settings.chat_max_concurrency)
    application.state.ingestion_semaphore = BoundedSemaphore(value=settings.upload_max_concurrency)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Educational mini RAG assistant for financial PDF documents.",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Liveness endpoint used by local checks and deployment probes."""
    return {"status": "ok"}
