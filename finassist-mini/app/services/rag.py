"""Orchestrates the complete Retrieval-Augmented Generation (RAG) flow."""

import logging

from app.models.document import RetrievedChunk
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.observability import current_memory_mb, timed_stage
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RAGService:
    """Runs embedding, retrieval, prompt-building, and generation as one workflow."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        llm_service: LLMService,
        top_k: int = 4,
        max_prompt_chars: int = 6000,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.llm_service = llm_service
        self.top_k = top_k
        self.max_prompt_chars = max_prompt_chars

    def answer_question(self, question: str) -> tuple[str, list[dict[str, int | str]]]:
        """Execute full RAG pipeline and return answer plus source citations."""
        with timed_stage() as embed_stage:
            query_embedding = self.embedding_service.embed_text(question)
        with timed_stage() as retrieve_stage:
            retrieved_chunks = self.vector_store_service.search(query_embedding=query_embedding, top_k=self.top_k)
        with timed_stage() as prompt_stage:
            prompt = self._build_prompt(question=question, retrieved_chunks=retrieved_chunks)
        with timed_stage() as llm_stage:
            answer = self.llm_service.generate_answer(prompt)
        logger.info(
            "chat_pipeline_metrics question_chars=%d retrieved_chunks=%d prompt_chars=%d embed_s=%.3f retrieve_s=%.3f prompt_s=%.3f llm_s=%.3f py_mem_mb=%.2f",
            len(question),
            len(retrieved_chunks),
            len(prompt),
            embed_stage[0],
            retrieve_stage[0],
            prompt_stage[0],
            llm_stage[0],
            current_memory_mb(),
        )
        return answer, self._build_sources(retrieved_chunks)

    def _build_prompt(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        """Compose a grounded prompt that restricts model output to retrieved context."""
        if not retrieved_chunks:
            context = "No relevant context was found in the uploaded documents."
        else:
            context_parts: list[str] = []
            current_size = 0
            for chunk in retrieved_chunks:
                chunk_text = (
                    f"[Document: {chunk.document_name} | Page: {chunk.page_number}]\n"
                    f"{chunk.text}"
                )
                projected = current_size + len(chunk_text) + (2 if context_parts else 0)
                if projected > self.max_prompt_chars:
                    break
                context_parts.append(chunk_text)
                current_size = projected
            context = "\n\n".join(context_parts) if context_parts else "Context was retrieved but exceeded prompt budget."

        return (
            "You are FinAssist, a financial assistant. "
            "Answer strictly from the provided context. "
            "If the answer is not in context, say you do not have enough information from uploaded documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def _build_sources(self, retrieved_chunks: list[RetrievedChunk]) -> list[dict[str, int | str]]:
        """Build unique source references for API output."""
        seen: set[tuple[str, int]] = set()
        sources: list[dict[str, int | str]] = []
        for chunk in retrieved_chunks:
            key = (chunk.document_name, chunk.page_number)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"document": chunk.document_name, "page": chunk.page_number})
        return sources
